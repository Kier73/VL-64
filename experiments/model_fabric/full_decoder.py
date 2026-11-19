"""Stream an observed Llama DAG across ALL layers and the complete vocabulary.

This optional VL-64 model-fabric adapter is inspired by E-CPU HFG-003 model
handoffs and VLD paged model-orbit checkpoints. The compact src/vld core is
unchanged. One layer's exact admitted weights are held at a time, not 30.

The trusted external checkpoint head is a REQUIRED restore argument. Digests
bind bytes but DO NOT authenticate remote hosts or form a consensus quorum.
A completed run performs local numeric operators using exact range-acquired
model values; this is not proof of execution on remote processors.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from vld import domain_digest, rms_norm
from .execution_dag import MAX_RANGE, QualifiedTensorReader
from .pretrained_layer import PreparedLayer, _verify_layer_nodes
from .pretrained_orbit import (
    ContentAddressedPages, _atomic_write, _encoded, _strict_json,
    _put_blob, _get_blob, _put_f64, _get_f64, _static_weight_identity,
)

SCHEMA = "vl64.full-pretrained-decoder-checkpoint.v1"
RESULT_SCHEMA = "vl64.full-pretrained-decoder-result.v1"
KV_SCHEMA = "vl64.full-pretrained-layer-kv.v1"
GENESIS = "0" * 64
MAX_TOKENS = 32


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _finite_vectors(vectors, width):
    if (not isinstance(vectors, list) or not vectors or len(vectors) > MAX_TOKENS or
            any(not isinstance(vec, list) or len(vec) != width or
                any(type(x) not in (float, int) or not math.isfinite(x)
                    for x in vec) for vec in vectors)):
        raise ValueError("Unqualified hidden-state matrix")
    return b"".join(struct.pack("<" + "d"*width, *vec) for vec in vectors)


def _put_hidden(store, vectors, width):
    return _put_blob(store, _finite_vectors(vectors, width))


def _get_hidden(store, manifest, count, width):
    raw = _get_blob(store, manifest, expected_bytes=count*width*8)
    values = struct.unpack("<" + "d"*(count*width), raw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite restored hidden-state coordinate")
    return [list(values[i*width:(i+1)*width]) for i in range(count)]


def _verify_checkpoint(plan, checkpoint, *, expected_head, tokens):
    a = plan["architecture"]
    fields = {"schema","model","revision","dag_digest","tensor_ground_digest",
              "tokens","completed_layers","hidden","kv_manifests","weight_identities",
              "range_witnesses","bytes_acquired","ranges_acquired","owner","epoch",
              "previous_checkpoint","transcript_head","checkpoint_digest"}
    if (not isinstance(checkpoint, dict) or set(checkpoint)!=fields or
            checkpoint["schema"]!=SCHEMA or checkpoint["model"]!=plan["model"] or
            checkpoint["revision"]!=plan["revision"] or
            checkpoint["dag_digest"]!=plan["dag_digest"] or
            checkpoint["tensor_ground_digest"]!=plan["tensor_ground_digest"] or
            checkpoint["checkpoint_digest"]!=expected_head or
            checkpoint["checkpoint_digest"]!=domain_digest(SCHEMA,{
                k:v for k,v in checkpoint.items() if k!="checkpoint_digest"}) or
            checkpoint["tokens"]!=list(tokens) or
            type(checkpoint["completed_layers"]) is not int or
            not 0<=checkpoint["completed_layers"]<=a["layers"] or
            type(checkpoint["epoch"]) is not int or checkpoint["epoch"]<0 or
            not isinstance(checkpoint["owner"],str) or not checkpoint["owner"] or
            not isinstance(checkpoint["kv_manifests"],list) or
            len(checkpoint["kv_manifests"])!=checkpoint["completed_layers"] or
            not isinstance(checkpoint["weight_identities"],dict) or
            set(checkpoint["weight_identities"])==set() and checkpoint["completed_layers"] != 0 or
            set(checkpoint["weight_identities"])!={str(i) for i in range(checkpoint["completed_layers"])} or
            set(checkpoint["range_witnesses"])!={str(i) for i in range(-1,checkpoint["completed_layers"])} or
            type(checkpoint["bytes_acquired"]) is not int or checkpoint["bytes_acquired"]<0 or
            type(checkpoint["ranges_acquired"]) is not int or checkpoint["ranges_acquired"]<0):
        raise ValueError("Checkpoint is not the externally admitted model frontier")
    for k in ("transcript_head","previous_checkpoint"):
        if not isinstance(checkpoint[k],str) or len(checkpoint[k])!=64:
            raise ValueError("Invalid checkpoint causal lineage")
    if any(not isinstance(v,str) or len(v)!=64 for v in checkpoint["kv_manifests"]):
        raise ValueError("Invalid KV manifest identity")
    if any(not isinstance(v,str) or len(v)!=64 for v in checkpoint["weight_identities"].values()):
        raise ValueError("Invalid layer static weight identity")
    _get_hidden  # Actual hidden/page check is performed before restoring state.
    return checkpoint["completed_layers"]


def _put_kv(store, plan, layer):
    a = plan["architecture"]
    width = a["kv_heads"]*a["head_dim"]
    if layer.step!=len(layer.keys) or layer.step!=len(layer.values):
        raise ValueError("Incomplete causal layer history")
    k = [[v for head in pos for v in head] for pos in layer.keys]
    v = [[value for head in pos for value in head] for pos in layer.values]
    record = {"schema": KV_SCHEMA, "dag_digest": plan["dag_digest"],
              "layer":layer.index,"positions":layer.step,
              "k":_put_blob(store,_finite_vectors(k,width)),
              "v":_put_blob(store,_finite_vectors(v,width))}
    raw = _encoded(record)
    if len(raw)>MAX_RANGE:
        raise ValueError("Causal KV manifest exceeds bounded message")
    return store.put(raw)


def _check_kv(store, plan, manifest, index, count):
    try:
        record = _strict_json(store.get(manifest))
    except (OSError, KeyError, ValueError) as e:
        raise ValueError("Missing or corrupt checkpoint KV page") from e
    if (not isinstance(record,dict) or set(record)!={
            "schema","dag_digest","layer","positions","k","v"} or
            record["schema"]!=KV_SCHEMA or record["dag_digest"]!=plan["dag_digest"] or
            record["layer"]!=index or record["positions"]!=count):
        raise ValueError("Checkpoint KV coordinate is inconsistent")
    width=plan["architecture"]["kv_heads"]*plan["architecture"]["head_dim"]
    return _get_hidden(store,record["k"],count,width),_get_hidden(store,record["v"],count,width)


class StreamingDecoder:
    """Layer-major prefill / reproducible full logits for supplied token IDs.

    Only local operator execution is claimed. `host_for(index)` returns an
    approved weight reader; index -1 is embeddings and index N is final head.
    Exact source policy (`approved_for(index)`) is supplied by the caller.
    """
    def __init__(self,plan,tokens,host_for,approved_for,store,*,
                 owner="local-executor",epoch=0,checkpoint=None,expected_head=None):
        a=plan["architecture"]
        if (not isinstance(tokens,(list,tuple)) or
                not 1<=len(tokens)<=min(a["context"],MAX_TOKENS) or
                any(type(x) is not int or not 0<=x<a["vocab"] for x in tokens) or
                not isinstance(store,ContentAddressedPages) or
                not callable(host_for) or not callable(approved_for) or
                not isinstance(owner,str) or not owner or
                type(epoch) is not int or epoch<0):
            raise ValueError("Unadmitted full-decoder execution request")
        self.plan,self.tokens,self.store=plan,list(tokens),store
        self.host_for,self.approved_for=host_for,approved_for
        self.owner,self.epoch=owner,epoch
        self.completed_layers=0
        self.kv_manifests=[]
        self.weight_identities={}
        self.range_witnesses={}
        self.bytes_acquired=0
        self.ranges_acquired=0
        self.previous_checkpoint=GENESIS
        self.transcript_head=GENESIS
        if checkpoint is not None:
            if expected_head is None or epoch<=checkpoint.get("epoch",-1):
                raise ValueError("Externally pinned newer epoch required for recovery")
            self.completed_layers=_verify_checkpoint(plan,checkpoint,
                expected_head=expected_head,tokens=tokens)
            self.hidden=_get_hidden(store,checkpoint["hidden"],len(tokens),a["hidden"])
            self.kv_manifests=list(checkpoint["kv_manifests"])
            for i,item in enumerate(self.kv_manifests):
                _check_kv(store,plan,item,i,len(tokens))
            self.weight_identities=dict(checkpoint["weight_identities"])
            self.range_witnesses=dict(checkpoint["range_witnesses"])
            self.bytes_acquired=checkpoint["bytes_acquired"]
            self.ranges_acquired=checkpoint["ranges_acquired"]
            self.previous_checkpoint=checkpoint["checkpoint_digest"]
            self.transcript_head=checkpoint["transcript_head"]
        else:
            if expected_head is not None:
                raise ValueError("No initial checkpoint to match expected head")
            reader=QualifiedTensorReader(plan,host_for(-1),
                                         approved_hosts=approved_for(-1))
            self.hidden=[reader.embedding(token) for token in tokens]
            self.bytes_acquired=reader.bytes_read
            self.ranges_acquired=reader.ranges
            self.range_witnesses["-1"]=reader.receipt_chain
        # This is an in-memory pending state until the caller publishes the
        # checkpoint and its externally protected expected head.
        self.hidden_manifest=_put_hidden(store,self.hidden,a["hidden"])

    def advance_layer(self):
        i=self.completed_layers
        a=self.plan["architecture"]
        if i>=a["layers"]:
            raise ValueError("All decoder layers already committed")
        # Validate the declared DAG for this layer BEFORE admitting weights.
        _verify_layer_nodes(self.plan,i)
        reader=QualifiedTensorReader(self.plan,self.host_for(i),
                                     approved_hosts=self.approved_for(i))
        layer=PreparedLayer(reader,i)
        outputs=[]
        for state in self.hidden:
            result,_=layer.next(state,trace=False)
            outputs.append(result)
        kv_digest=_put_kv(self.store,self.plan,layer)
        next_manifest=_put_hidden(self.store,outputs,a["hidden"])
        identity=_static_weight_identity(self.plan,layer)
        # Only the fully qualified layer frontier becomes visible.
        new_head=domain_digest("vl64.full-layer-transition.v1",{
            "previous":self.transcript_head,"dag_digest":self.plan["dag_digest"],
            "layer":i,"tokens":self.tokens,"hidden_sha256":next_manifest["sha256"],
            "kv_manifest":kv_digest,"static_weight_identity":identity})
        self.completed_layers+=1
        self.kv_manifests.append(kv_digest)
        self.hidden=outputs
        self.hidden_manifest=next_manifest
        self.weight_identities[str(i)]=identity
        self.range_witnesses[str(i)]=reader.receipt_chain
        self.bytes_acquired+=reader.bytes_read
        self.ranges_acquired+=reader.ranges
        self.transcript_head=new_head
        return {"completed_layers":self.completed_layers,
                "last_layer":i,"hidden_sha256":next_manifest["sha256"],
                "static_weight_identity":identity,"causal_head":new_head,
                "layer_weight_bytes":reader.bytes_read,"layer_range_reads":reader.ranges}

    def checkpoint(self):
        snapshot={"schema":SCHEMA,"model":self.plan["model"],
                  "revision":self.plan["revision"],
                  "dag_digest":self.plan["dag_digest"],
                  "tensor_ground_digest":self.plan["tensor_ground_digest"],
                  "tokens":list(self.tokens),"completed_layers":self.completed_layers,
                  "hidden":self.hidden_manifest,"kv_manifests":list(self.kv_manifests),
                  "weight_identities":dict(self.weight_identities),
                  "range_witnesses":dict(self.range_witnesses),
                  "bytes_acquired":self.bytes_acquired,
                  "ranges_acquired":self.ranges_acquired,
                  "owner":self.owner,"epoch":self.epoch,
                  "previous_checkpoint":self.previous_checkpoint,
                  "transcript_head":self.transcript_head}
        snapshot["checkpoint_digest"]=domain_digest(SCHEMA,snapshot)
        if len(_encoded(snapshot))>MAX_RANGE:
            raise ValueError("Decoder checkpoint exceeds bounded message")
        return snapshot

    def logits(self,*,top_n=5,return_logits=False):
        a=self.plan["architecture"]
        if self.completed_layers!=a["layers"]:
            raise ValueError("Vocabulary projection requires every declared layer")
        if type(top_n) is not int or not 1<=top_n<=min(32,a["vocab"]):
            raise ValueError("Invalid vocabulary output width")
        # Fail closed if final typed edges were substituted and re-digested.
        end=f"layer.{a['layers']-1}.ffn_residual"
        terminal=self.plan["nodes"][-2:]
        projection="model.embed_tokens.weight" if a["tied_embedding"] else "lm_head.weight"
        if terminal!=[
                {"id":"final_norm","op":"rms_norm","inputs":[end],
                 "output_shape":[a["hidden"]],"weight":"model.norm.weight"},
                {"id":"vocab_logits","op":"linear","inputs":["final_norm"],
                 "output_shape":[a["vocab"]],"weight":projection}]:
            raise ValueError("Unadmitted final norm/vocabulary DAG contract")
        reader=QualifiedTensorReader(self.plan,self.host_for(a["layers"]),
                                     approved_hosts=self.approved_for(a["layers"]))
        gamma=reader.values("model.norm.weight")
        normalized=rms_norm(self.hidden[-1],gamma,eps=a["eps"])
        logits=reader.matrix_vec(projection,normalized)
        if len(logits)!=a["vocab"] or not all(math.isfinite(x) for x in logits):
            raise ValueError("Incomplete/nonfinite pretrained vocabulary output")
        raw=struct.pack("<"+"d"*len(logits),*logits)
        best=sorted(range(len(logits)),key=lambda i:(-logits[i],i))[:top_n]
        receipt={"schema":RESULT_SCHEMA,"status":"FULL_WEIGHT_BACKED_DECODER_FORWARD",
                 "model":self.plan["model"],"revision":self.plan["revision"],
                 "dag_digest":self.plan["dag_digest"],
                 "tokens":list(self.tokens),"layers_executed":self.completed_layers,
                 "checkpoint_digest":self.checkpoint()["checkpoint_digest"],
                 "full_vocab_size":len(logits),"logits_f64_sha256":_sha(raw),
                 "top_token_ids":best,"top_logits":[logits[i] for i in best],
                 "selected_next_token":best[0],
                 "weight_payload_bytes_acquired":self.bytes_acquired+reader.bytes_read,
                 "exact_range_count":self.ranges_acquired+reader.ranges,
                 "projection_witness_chain":reader.receipt_chain,
                 "scope":"provided_ids_forward_pass_no_tokenizer_or_remote_compute_attestation",
                 "authenticity_scope":"approved_range_hosts_digest_checked_no_full_model_signature"}
        receipt["receipt_digest"]=domain_digest(RESULT_SCHEMA,receipt)
        return (receipt,logits) if return_logits else receipt


def save_full_checkpoint(path,snapshot):
    if (not isinstance(snapshot,dict) or snapshot.get("schema")!=SCHEMA or
            snapshot.get("checkpoint_digest")!=domain_digest(SCHEMA,{
                k:v for k,v in snapshot.items() if k!="checkpoint_digest"})):
        raise ValueError("Unverified full-decoder checkpoint cannot be published")
    raw=_encoded(snapshot)
    if len(raw)>MAX_RANGE:
        raise ValueError("Oversized full-decoder checkpoint")
    _atomic_write(path,raw)
    return snapshot["checkpoint_digest"]


def load_full_checkpoint(path):
    from pathlib import Path
    with Path(path).open("rb") as source:
        raw=source.read(MAX_RANGE+1)
    if not 0<len(raw)<=MAX_RANGE:
        raise ValueError("Missing/bounded full-decoder checkpoint")
    result=_strict_json(raw)
    if not isinstance(result,dict) or result.get("schema")!=SCHEMA:
        raise ValueError("Unsupported full-decoder checkpoint")
    return result
