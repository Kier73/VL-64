"""One complete reusable pretrained Llama layer, admitted by observer-host ranges.

The manifest describes computation but never supplies tensor values. A qualified
host must provide every exact BF16/F32 range. We reuse nine admitted matrices
across positions and retain an order-dependent evidence chain; no full language
model or next-token claim follows from this layer's output.
"""
from __future__ import annotations

import hashlib
import math
import struct

from vld import StructuralLinearOperator, domain_digest, rms_norm
from .execution_dag import (QualifiedTensorReader, apply_rope,
                            grouped_query_attention, swiglu)

LAYER_TENSORS = (
    "input_layernorm.weight", "self_attn.q_proj.weight",
    "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
STAGES = (
    "input_norm", "q", "k", "v", "q_rope", "k_rope", "attention",
    "o", "attention_residual", "post_norm", "gate", "up",
    "swiglu", "down", "ffn_residual",
)
MAX_POSITIONS = 32  # bounded experiment, not the model context limit


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fingerprint(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite or empty intermediate layer value")
    return {"elements": len(values),
            "f32_sha256": _sha(b"".join(struct.pack("<f",float(x)) for x in values)),
            "first_4": [round(x,8) for x in values[:4]],
            "rms": round(math.sqrt(math.fsum(x*x for x in values)/len(values)),8)}


def _verify_layer_nodes(plan,index):
    schema = "vl64.typed-pretrained-execution-dag.v1"
    if (not isinstance(plan,dict) or plan.get("schema") != schema or
            plan.get("dag_digest") != domain_digest(
                schema,{k:v for k,v in plan.items() if k!="dag_digest"})):
        raise ValueError("Unverified typed model execution DAG")
    arch = plan["architecture"]
    if type(index) is not int or not 0 <= index < arch["layers"]:
        raise ValueError("Unadmitted transformer layer index")
    nodes = {node["id"]: node for node in plan["nodes"]}
    prefix = f"layer.{index}."
    if not all(prefix+stage in nodes for stage in STAGES):
        raise ValueError("Incomplete typed layer execution DAG")
    ops = ("rms_norm","linear","linear","linear","rope","rope",
           "causal_gqa","linear","residual","rms_norm","linear",
           "linear","silu_mul","linear","residual")
    if [nodes[prefix+s]["op"] for s in STAGES] != list(ops):
        raise ValueError("Typed DAG stage differs from executable decoder")
    base = f"model.layers.{index}."
    for suffix in LAYER_TENSORS:
        if base+suffix not in plan["tensors"]:
            raise ValueError("Unadmitted transformer layer tensor: "+suffix)
    return arch


def _matvec(flat,rows,cols,x):
    """Reuse VL64's linear operator in bounded row windows."""
    if len(x)!=cols or len(flat)!=rows*cols:
        raise ValueError("Admitted tensor shape differs from linear operation")
    out=[]
    batch=max(1,8192//cols)
    for first in range(0,rows,batch):
        n=min(batch,rows-first)
        sparse={(r,c):flat[(first+r)*cols+c] for r in range(n)
                for c in range(cols) if flat[(first+r)*cols+c]!=0.0}
        out.extend(StructuralLinearOperator(rows=n,cols=cols,
                                            sparse_delta=sparse).matvec(x))
    if not all(math.isfinite(v) for v in out):
        raise ValueError("Nonfinite pretrained linear output")
    return out


class PreparedLayer:
    """Nine once-admitted tensors plus a causal, position-local KV cache."""
    def __init__(self,reader,index=0):
        self.reader=reader
        self.arch=_verify_layer_nodes(reader.plan,index)
        self.index=index
        prefix=f"model.layers.{index}."
        self.weights={}
        for suffix in LAYER_TENSORS:
            self.weights[suffix]=reader.values(prefix+suffix)
        self.keys=[]
        self.values=[]
        self.step=0
        self.weight_range_chain=reader.receipt_chain
        self.weight_bytes=reader.bytes_read

    def _linear(self,suffix,x):
        name=f"model.layers.{self.index}.{suffix}"
        rows,cols=self.reader.plan["tensors"][name]["shape"]
        return _matvec(self.weights[suffix],rows,cols,x)

    def next(self,hidden,*,trace=True):
        a=self.arch
        if len(hidden)!=a["hidden"] or self.step>=a["context"]:
            raise ValueError("Unadmitted hidden vector or position")
        pos=self.step
        d=a["head_dim"]
        outputs={}
        def stage(name,values):
            if trace:
                outputs[f"layer.{self.index}.{name}"]=_fingerprint(values)
            return values
        norm=stage("input_norm",rms_norm(
            hidden,self.weights["input_layernorm.weight"],eps=a["eps"]))
        q=stage("q",self._linear("self_attn.q_proj.weight",norm))
        k=stage("k",self._linear("self_attn.k_proj.weight",norm))
        v=stage("v",self._linear("self_attn.v_proj.weight",norm))
        qh=[apply_rope(q[j*d:(j+1)*d],pos,a["theta"])
            for j in range(a["heads"])]
        kh=[apply_rope(k[j*d:(j+1)*d],pos,a["theta"])
            for j in range(a["kv_heads"])]
        vh=[v[j*d:(j+1)*d] for j in range(a["kv_heads"])]
        stage("q_rope",[z for head in qh for z in head])
        stage("k_rope",[z for head in kh for z in head])
        self.keys.append(kh)
        self.values.append(vh)
        context=stage("attention",grouped_query_attention(
            qh,self.keys,self.values,heads=a["heads"],kv_heads=a["kv_heads"]))
        attn=stage("o",self._linear("self_attn.o_proj.weight",context))
        residual=stage("attention_residual",[x+y for x,y in zip(hidden,attn)])
        post=stage("post_norm",rms_norm(
            residual,self.weights["post_attention_layernorm.weight"],eps=a["eps"]))
        gate=stage("gate",self._linear("mlp.gate_proj.weight",post))
        up=stage("up",self._linear("mlp.up_proj.weight",post))
        intermediate=stage("swiglu",swiglu(gate,up))
        down=stage("down",self._linear("mlp.down_proj.weight",intermediate))
        result=stage("ffn_residual",[x+y for x,y in zip(residual,down)])
        self.step+=1
        return result,outputs


def execute_pretrained_layer(plan,tokens,observer_host,*,approved_hosts,
                             layer=0,include_output=True):
    """Execute a complete weight-backed layer, not final vocabulary logits."""
    a=plan["architecture"]
    if (not isinstance(tokens,(list,tuple)) or
            not 1<=len(tokens)<=min(a["context"],MAX_POSITIONS) or
            any(type(t) is not int or not 0<=t<a["vocab"] for t in tokens)):
        raise ValueError("Unqualified token sequence")
    reader=QualifiedTensorReader(plan,observer_host,approved_hosts=approved_hosts)
    program=PreparedLayer(reader,layer)
    stages=[]
    for position,token in enumerate(tokens):
        embedding=reader.embedding(token)
        result,trace=program.next(embedding)
        stages.append({"position":position,"token_id":token,
                       "embedding":_fingerprint(embedding),"stages":trace})
    receipt={"schema":"vl64.pretrained-transformer-layer-receipt.v1",
             "status":"WEIGHT_BACKED_PRETRAINED_LAYER_EXECUTED",
             "model":plan["model"],"revision":plan["revision"],
             "dag_digest":plan["dag_digest"],
             "tensor_ground_digest":plan["tensor_ground_digest"],
             "layer_index":layer,"tokens":list(tokens),"positions":stages,
             "weight_payload_bytes_acquired":reader.bytes_read,
             "exact_range_count":reader.ranges,
             "range_witness_chain_digest":reader.receipt_chain,
             "output":result if include_output else _fingerprint(result),
             "scope":"one_verified_pretrained_decoder_layer_not_full_model_logits",
             "authenticity_scope":"host_approved_and_byte_hash_checked; no_full_artifact_digest"}
    receipt["receipt_digest"]=domain_digest(receipt["schema"],receipt)
    return receipt
