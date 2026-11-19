"""Crash-resumable pretrained causal computation with exact paged KV state.

Research bridge to E-CPU HFG-003/M1 recovery and Virtual-Layer-Dynamics
PagedKVCacheGround / scheduler-scoped temporal coordinates. The layer math
is the existing VL-64 PreparedLayer; this module supplies content-addressed
state, externally pinned recovery lineage and independent range-host rehome.

SHA-256 is integrity relative to an external expected checkpoint head, NOT
host authentication, quorum consensus, or remote processor attestation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path

from vld import domain_digest
from .execution_dag import MAX_RANGE, QualifiedTensorReader
from .pretrained_chain import GENESIS, admit_handoff, make_handoff
from .pretrained_layer import MAX_POSITIONS, PreparedLayer, LAYER_TENSORS, _fingerprint

SCHEMA = "vl64.model-orbit-checkpoint.v2"
LAYER_SCHEMA = "vl64.model-orbit-layer-pages.v1"
STEP_SCHEMA = "vl64.model-orbit-step.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError("Expected a lowercase SHA-256 content digest")
    return value


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _strict_json(raw):
    def no_duplicates(pairs):
        result = {}
        for key, val in pairs:
            if key in result:
                raise ValueError("Duplicate checkpoint JSON field")
            result[key] = val
        return result
    try:
        return json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeError, TypeError) as exc:
        raise ValueError("Invalid checkpoint JSON") from exc


def _finite_f64(vector, width):
    if (not isinstance(vector, (tuple, list)) or len(vector) != width or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in vector)):
        raise ValueError("Invalid finite model-state coordinates")
    return struct.pack("<" + "d"*width, *vector)


def _atomic_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".vl64-state-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class ContentAddressedPages:
    """Single-writer atomic byte-page store. No distributed fencing is implied."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data):
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_RANGE:
            raise ValueError("Unqualified bounded model state page")
        digest = _sha(data)
        target = self.root / digest
        if target.exists():
            if self.get(digest) != data:
                raise ValueError("Conflicting content-addressed state page")
        else:
            _atomic_write(target, data)
        return digest

    def get(self, digest):
        target = self.root / _digest(digest)
        with target.open("rb") as stream:
            raw = stream.read(MAX_RANGE+1)
        if not 0 < len(raw) <= MAX_RANGE or _sha(raw) != digest:
            raise ValueError("State page missing valid content identity")
        return raw


def _put_blob(store, raw):
    """Split any state object into <=64KiB physically addressable pages."""
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Empty model state object")
    return {"encoding":"exact-bytes-v1", "bytes":len(raw), "sha256":_sha(raw),
            "pages":[store.put(raw[i:i+MAX_RANGE]) for i in range(0,len(raw),MAX_RANGE)]}


def _get_blob(store, manifest, *, expected_bytes):
    if (not isinstance(manifest, dict) or set(manifest) !=
            {"encoding", "bytes", "sha256", "pages"} or
            manifest["encoding"] != "exact-bytes-v1" or
            type(expected_bytes) is not int or expected_bytes <= 0 or
            type(manifest["bytes"]) is not int or manifest["bytes"] != expected_bytes or
            not isinstance(manifest["pages"], list) or
            len(manifest["pages"]) != math.ceil(expected_bytes/MAX_RANGE)):
        raise ValueError("State object manifest extent mismatch")
    _digest(manifest["sha256"])
    raw = b"".join(store.get(page) for page in manifest["pages"])
    if len(raw) != expected_bytes or _sha(raw) != manifest["sha256"]:
        raise ValueError("State object digest or length mismatch")
    return raw


def _put_f64(store, values, width):
    return _put_blob(store, _finite_f64(values, width))


def _get_f64(store, manifest, width):
    raw = _get_blob(store, manifest, expected_bytes=width*8)
    values = list(struct.unpack("<"+"d"*width, raw))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite restored model state")
    return values


def _static_weight_identity(plan, layer):
    """Host-independent identity of the ACTUAL admitted decoder-layer values.

    Different source host IDs change receipt provenance, not weight identity.
    SHA-256 is still content integrity, not a source signature. Check the
    literal IEEE754-F64 representation to avoid lossy checkpoint hashes.
    """
    records=[]
    for suffix in LAYER_TENSORS:
        name=f"model.layers.{layer.index}.{suffix}"
        values=layer.weights[suffix]
        extent=plan["tensors"][name]["shape"]
        if len(values)!=math.prod(extent):
            raise ValueError("Decoded pretrained tensor does not match the admitted shape")
        h=hashlib.sha256()
        for index in range(0,len(values),8192):
            batch=values[index:index+8192]
            h.update(struct.pack("<"+"d"*len(batch),*batch))
        records.append({"tensor":name,"shape":extent,"values_sha256":h.hexdigest()})
    return domain_digest("vl64.static-weight-content-identity.v1",{
        "dag_digest":plan["dag_digest"],"layer":layer.index,"tensors":records})


def _check_root(plan, snapshot, *, expected_head):
    if (not isinstance(snapshot, dict) or set(snapshot) != {
            "schema", "model", "revision", "dag_digest", "tensor_ground_digest",
            "layers_executed", "tokens", "epoch", "owner", "previous_checkpoint",
            "last_handoff_digest", "transcript_head", "last_hidden", "layer_manifests",
            "weight_witness_chains", "static_weight_identities", "checkpoint_digest"} or
            snapshot["schema"] != SCHEMA or snapshot["model"] != plan["model"] or
            snapshot["revision"] != plan["revision"] or
            snapshot["dag_digest"] != plan["dag_digest"] or
            snapshot["tensor_ground_digest"] != plan["tensor_ground_digest"] or
            snapshot["checkpoint_digest"] != _digest(expected_head) or
            snapshot["checkpoint_digest"] != domain_digest(SCHEMA, {
                k:v for k,v in snapshot.items() if k != "checkpoint_digest"})):
        raise ValueError("Checkpoint identity or externally pinned head mismatch")
    a = plan["architecture"]
    count = len(snapshot["tokens"])
    if (not 1 <= count <= min(MAX_POSITIONS, a["context"]) or
            any(type(t) is not int or not 0 <= t < a["vocab"] for t in snapshot["tokens"]) or
            type(snapshot["epoch"]) is not int or snapshot["epoch"] < 0 or
            not isinstance(snapshot["owner"], str) or not snapshot["owner"] or
            type(snapshot["layers_executed"]) is not int or
            not 1 <= snapshot["layers_executed"] < a["layers"] or
            not isinstance(snapshot["layer_manifests"], list) or
            len(snapshot["layer_manifests"]) != snapshot["layers_executed"]+1 or
            not isinstance(snapshot["weight_witness_chains"], dict) or
            not isinstance(snapshot["static_weight_identities"], dict) or
            set(snapshot["static_weight_identities"]) != {str(i) for i in range(snapshot["layers_executed"]+1)} or
            set(snapshot["weight_witness_chains"]) != {
                str(i) for i in range(-1, snapshot["layers_executed"]+1)}):
        raise ValueError("Invalid causal model checkpoint scope")
    for field in ("last_handoff_digest", "transcript_head", "previous_checkpoint"):
        _digest(snapshot[field])
    for value in snapshot["layer_manifests"]:
        _digest(value)
    for value in snapshot["weight_witness_chains"].values():
        _digest(value)
    for value in snapshot["static_weight_identities"].values():
        _digest(value)
    return count


class RecoverableModelOrbit:
    """Monotonic token-prefix session; only accepted full positions are committed."""

    def __init__(self, plan, range_hosts, *, approved_hosts, compute_actors,
                 approved_producers, store, owner="local-observer", epoch=0,
                 last_layer=1, transfer=None):
        a = plan["architecture"]
        if (type(last_layer) is not int or not 1 <= last_layer < a["layers"] or
                type(epoch) is not int or epoch < 0 or
                not isinstance(owner, str) or not owner or
                not isinstance(range_hosts, dict) or not isinstance(approved_hosts, dict) or
                not isinstance(compute_actors, dict) or
                not isinstance(approved_producers, (set, tuple, list)) or
                not isinstance(store, ContentAddressedPages) or
                (transfer is not None and not callable(transfer))):
            raise ValueError("Unqualified model-orbit assignment")
        for index in range(-1, last_layer+1):
            if index not in range_hosts or index not in approved_hosts:
                raise ValueError("Missing qualified model-orbit weight host")
        for index in range(last_layer):
            if compute_actors.get(index) not in approved_producers:
                raise ValueError("Unapproved model-orbit handoff actor")
        self.plan = plan
        self.store = store
        self.owner, self.epoch, self.last_layer = owner, epoch, last_layer
        self.compute_actors, self.approved_producers = dict(compute_actors), set(approved_producers)
        self.transfer = transfer
        self.readers = {i:QualifiedTensorReader(plan,range_hosts[i],
                        approved_hosts=approved_hosts[i]) for i in range(-1,last_layer+1)}
        self.layers = [PreparedLayer(self.readers[i],i) for i in range(last_layer+1)]
        self.static_weight_identities={str(i):_static_weight_identity(plan,layer)
                                       for i,layer in enumerate(self.layers)}
        self.tokens = []
        self.kv_refs = [[] for _ in self.layers]
        self.previous_checkpoint = GENESIS
        self.last_handoff = GENESIS
        self.transcript_head = GENESIS
        self.last_hidden = None

    def advance(self, token, *, expected_position=None):
        a = self.plan["architecture"]
        position = len(self.tokens)
        if (type(token) is not int or not 0 <= token < a["vocab"] or
                position >= min(MAX_POSITIONS,a["context"]) or
                (expected_position is not None and expected_position != position)):
            raise ValueError("Unqualified, replayed, or out-of-order token position")
        before = [layer.step for layer in self.layers]
        prior_readers = {i:(r.bytes_read,r.ranges,r.receipt_chain)
                         for i,r in self.readers.items()}
        previous = self.last_handoff
        pending = []
        try:
            state = self.readers[-1].embedding(token)
            for index, layer in enumerate(self.layers):
                state, _ = layer.next(state,trace=False)
                if index < self.last_layer:
                    packet = make_handoff(self.plan,index,position,state,
                        producer=self.compute_actors[index],
                        range_chain=self.readers[index].receipt_chain,preceding=previous)
                    received = self.transfer(dict(packet)) if self.transfer else packet
                    next_state = admit_handoff(self.plan,received,index=index,position=position,
                        preceding=previous,approved_producers=self.approved_producers,
                        expected_vector=state)
                    if received["source_weight_range_chain"] != self.readers[index].receipt_chain:
                        raise ValueError("State handoff lost source weight witness")
                    state = next_state
                    previous = received["handoff_digest"]
                    pending.append(previous)
            # Prepare durable pages before advancing the published causal head.
            new_refs = []
            d = a["kv_heads"] * a["head_dim"]
            for layer in self.layers:
                k = [x for head in layer.keys[-1] for x in head]
                v = [x for head in layer.values[-1] for x in head]
                new_refs.append({"k":_put_f64(self.store,k,d),
                                 "v":_put_f64(self.store,v,d)})
            final = _put_f64(self.store,state,a["hidden"])
            output = _fingerprint(state)
            head = domain_digest(STEP_SCHEMA,{
                "previous":self.transcript_head, "epoch":self.epoch,
                "owner":self.owner,"position":position,"token":token,
                "dag_digest":self.plan["dag_digest"],"handoffs":pending,
                "output_sha256":final["sha256"]})
        except Exception:
            for i,reader in self.readers.items():
                reader.bytes_read,reader.ranges,reader.receipt_chain = prior_readers[i]
            for layer,step in zip(self.layers,before):
                layer.keys=layer.keys[:step]
                layer.values=layer.values[:step]
                layer.step=step
            raise
        for refs,pair in zip(self.kv_refs,new_refs):
            refs.append(pair)
        self.tokens.append(token)
        self.last_hidden=final
        self.last_handoff=previous
        self.transcript_head=head
        return {"position":position,"token_id":token,"final_hidden":output,
                "handoff_digests":pending,"transcript_head":head,"state":state}

    def checkpoint(self):
        if not self.tokens or any(layer.step!=len(self.tokens) for layer in self.layers):
            raise ValueError("No consistent model-orbit causal frontier")
        manifests = []
        for index,refs in enumerate(self.kv_refs):
            body = {"schema":LAYER_SCHEMA,"dag_digest":self.plan["dag_digest"],
                    "layer":index,"positions":len(self.tokens),"kv_refs":refs}
            encoded = _encoded(body)
            if len(encoded)>MAX_RANGE:
                raise ValueError("Layer state manifest exceeds bounded byte frame")
            manifests.append(self.store.put(encoded))
        root = {"schema":SCHEMA,"model":self.plan["model"],
                "revision":self.plan["revision"],"dag_digest":self.plan["dag_digest"],
                "tensor_ground_digest":self.plan["tensor_ground_digest"],
                "layers_executed":self.last_layer,"tokens":list(self.tokens),
                "epoch":self.epoch,"owner":self.owner,
                "previous_checkpoint":self.previous_checkpoint,
                "last_handoff_digest":self.last_handoff,
                "transcript_head":self.transcript_head,
                "last_hidden":self.last_hidden,"layer_manifests":manifests,
                "weight_witness_chains":{str(i):r.receipt_chain
                                         for i,r in self.readers.items()},
                "static_weight_identities":dict(self.static_weight_identities)}
        root["checkpoint_digest"] = domain_digest(SCHEMA,root)
        if len(_encoded(root))>MAX_RANGE:
            raise ValueError("Root checkpoint exceeds bounded byte frame")
        return root

    @classmethod
    def restore(cls, plan, snapshot, *, expected_head, new_owner, new_epoch,
                range_hosts, approved_hosts, compute_actors,approved_producers,
                store, transfer=None):
        count = _check_root(plan,snapshot,expected_head=expected_head)
        if (type(new_epoch) is not int or new_epoch<=snapshot["epoch"] or
                not isinstance(new_owner,str) or not new_owner):
            raise ValueError("Stale model-orbit assignment epoch or owner")
        if not isinstance(store,ContentAddressedPages):
            raise ValueError("No content-addressed model-state storage")
        a = plan["architecture"]
        width = a["kv_heads"]*a["head_dim"]
        loaded = []
        for index, digest in enumerate(snapshot["layer_manifests"]):
            record = _strict_json(store.get(digest))
            if (not isinstance(record,dict) or set(record) !=
                    {"schema","dag_digest","layer","positions","kv_refs"} or
                    record["schema"] != LAYER_SCHEMA or
                    record["dag_digest"] != plan["dag_digest"] or
                    record["layer"] != index or record["positions"] != count or
                    not isinstance(record["kv_refs"],list) or
                    len(record["kv_refs"]) != count):
                raise ValueError("Layer KV state manifest fails causal coordinate")
            history_k,history_v = [],[]
            for pair in record["kv_refs"]:
                if not isinstance(pair,dict) or set(pair)!={"k","v"}:
                    raise ValueError("Malformed KV page pair")
                k = _get_f64(store,pair["k"],width)
                v = _get_f64(store,pair["v"],width)
                history_k.append([k[j*a["head_dim"]:(j+1)*a["head_dim"]]
                                  for j in range(a["kv_heads"])])
                history_v.append([v[j*a["head_dim"]:(j+1)*a["head_dim"]]
                                  for j in range(a["kv_heads"])])
            loaded.append((record["kv_refs"],history_k,history_v))
        last = _get_f64(store,snapshot["last_hidden"],a["hidden"])
        session = cls(plan,range_hosts,approved_hosts=approved_hosts,
                      compute_actors=compute_actors,approved_producers=approved_producers,
                      store=store,owner=new_owner,epoch=new_epoch,
                      last_layer=snapshot["layers_executed"],transfer=transfer)
        if session.static_weight_identities != snapshot["static_weight_identities"]:
            raise ValueError("Rehomed observer supplied a different static weight snapshot")
        session.tokens=list(snapshot["tokens"])
        session.previous_checkpoint=snapshot["checkpoint_digest"]
        session.last_handoff=snapshot["last_handoff_digest"]
        session.transcript_head=snapshot["transcript_head"]
        session.last_hidden=snapshot["last_hidden"]
        for layer,(refs,k,v) in zip(session.layers,loaded):
            layer.keys,layer.values=k,v
            layer.step=count
        session.kv_refs=[refs for refs,_,_ in loaded]
        return session,last


def save_checkpoint(path, snapshot):
    """Atomic single-writer root publication; caller stores trusted head apart."""
    if not isinstance(snapshot,dict) or snapshot.get("schema")!=SCHEMA or \
            snapshot.get("checkpoint_digest")!=domain_digest(SCHEMA,{
                k:v for k,v in snapshot.items() if k!="checkpoint_digest"}):
        raise ValueError("Cannot publish invalid model-orbit checkpoint")
    raw=_encoded(snapshot)
    if len(raw)>MAX_RANGE:
        raise ValueError("Oversized model-orbit checkpoint")
    _atomic_write(path,raw)
    return snapshot["checkpoint_digest"]


def load_checkpoint(path):
    with Path(path).open("rb") as handle:
        raw=handle.read(MAX_RANGE+1)
    if not 0<len(raw)<=MAX_RANGE:
        raise ValueError("Oversized or empty checkpoint file")
    document=_strict_json(raw)
    if not isinstance(document,dict) or document.get("schema")!=SCHEMA:
        raise ValueError("Unknown checkpoint format")
    return document
