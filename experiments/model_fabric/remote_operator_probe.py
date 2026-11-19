"""Execute a *real pretrained model sub-operator* from HTTP ranged weights.

The full Hugging Face checkpoint is never downloaded. We read the pinned
tensor header, one BF16 layer-norm weight vector, and only 4 rows of the first
Q projection. The host supplies a deterministic test vector (not a real token
embedding). VL-64's rms_norm and StructuralLinearOperator perform the actual
numeric work; an independently written scalar reference checks the outputs.

This is exactly a selected pretrained weight-bearing *subgraph*, NOT a full
language model inference pass or reconstruction of missing weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from vld import StructuralLinearOperator,rms_norm,domain_digest
from .hf_glance import (
    MODEL,REVISION,_read,_url,observe_model,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bf16(data):
    if not isinstance(data,bytes) or len(data)%2:
        raise ValueError("Invalid BF16 selected tensor payload")
    result=[]
    for (bits,) in struct.iter_unpack("<H",data):
        val=struct.unpack("<f",struct.pack("<I",bits<<16))[0]
        if not math.isfinite(val):
            raise ValueError("Nonfinite observed model weight")
        result.append(val)
    return result


def _tensor(ground,name,shape):
    manifest=ground.get("tensors") or {}
    if manifest.get("tensor_count",0)>4096:
        raise ValueError("Oversized remote tensor manifest")
    hits=[t for t in manifest.get("tensors",[]) if t["name"]==name]
    if len(hits)!=1 or hits[0]["dtype"]!="BF16" or hits[0]["shape"]!=shape:
        raise ValueError("Unqualified BF16 tensor shape or name")
    return hits[0]


def execute_remote_q_slice(*,observed=None,model=MODEL,revision=REVISION):
    remote=observed if observed is not None else observe_model(
        model=model,revision=revision)
    if remote.get("tensor_header_status")!="OBSERVED_COMPLETE_TENSOR_HEADER":
        raise ValueError("Cannot execute without actual verified tensor header")
    if (remote["model"]!=model or remote["pinned_revision"]!=revision or
            remote["architecture_fields"].get("model_type")!="llama" or
            remote["architecture_fields"].get("attention_bias") not in (None,False)):
        raise ValueError("Unrecognized or unsupported remote model family")
    architecture=remote["architecture_fields"]
    hidden=architecture.get("hidden_size")
    epsilon=architecture.get("rms_norm_eps")
    if (type(hidden) is not int or not 16<=hidden<=8192 or
            type(epsilon) not in (float,int) or
            not 0<float(epsilon)<=1e-2):
        raise ValueError("No exact admissible remote normalization contract")
    first="model.layers.0.input_layernorm.weight"
    second="model.layers.0.self_attn.q_proj.weight"
    gamma_entry=_tensor(remote,first,[hidden])
    projection_entry=_tensor(remote,second,[hidden,hidden])
    row_count=4
    if row_count>hidden or row_count*hidden*2>16384:
        raise ValueError("Unqualified Q slice budget")
    file_prefix=remote["tensors"]["header_bytes"]
    weights_url=_url(model,revision,"model.safetensors")

    def ranged(entry,bytes_required):
        begin=file_prefix+entry["data_offsets"][0]
        last=begin+bytes_required-1
        content,receipt=_read(weights_url,start=begin,stop=last,
                              limit=bytes_required)
        if len(content)!=bytes_required:
            raise ValueError("Truncated pretrained weight slice")
        return content,{
            "tensor":entry["name"],"byte_begin":begin,
            "byte_end_exclusive":last+1,"sha256":_sha(content),
            "bytes_observed":len(content),
            "http_content_range":receipt["content_range"],
            "http_status":receipt["http_status"],
            "final_hostname":receipt["final_hostname"],
        }

    norm_bytes,norm_witness=ranged(gamma_entry,hidden*2)
    q_bytes,q_witness=ranged(projection_entry,row_count*hidden*2)
    gamma=_bf16(norm_bytes)
    qweights=_bf16(q_bytes)
    if len(gamma)!=hidden or len(qweights)!=row_count*hidden:
        raise ValueError("Inadequate actual BF16 tensor bytes")

    # Input is a user-independent numerical probe; no token identity claimed.
    input_vector=[
        struct.unpack("<f",struct.pack("<f",math.sin((i+1)*0.07)*0.25))[0]
        for i in range(hidden)]
    normalized=rms_norm(input_vector,gamma,eps=float(epsilon))
    operator=StructuralLinearOperator(
        rows=row_count,cols=hidden,
        sparse_delta={
            (r,c):qweights[r*hidden+c]
            for r in range(row_count) for c in range(hidden)
            if qweights[r*hidden+c]!=0
        })
    computed=operator.matvec(normalized)

    # Independent scalar reference doesn't call either VL-64 operator.
    scale=1/math.sqrt(
        sum(v*v for v in input_vector)/hidden + float(epsilon))
    reference_normalized=[
        input_vector[c]*scale*gamma[c] for c in range(hidden)]
    reference=[
        sum(qweights[r*hidden+c]*reference_normalized[c]
            for c in range(hidden))
        for r in range(row_count)]
    residual=max(abs(a-b) for a,b in zip(computed,reference))
    if residual>1e-7:
        raise RuntimeError("Remote BF16 operator disagrees with scalar reference")

    full_payload=remote["tensors"]["declared_payload_bytes"]
    payload=len(norm_bytes)+len(q_bytes)
    receipt={
        "schema":"vl64.real-pretrained-model-subgraph-execution.v1",
        "status":"ACTUAL_REMOTE_PRETRAINED_WEIGHT_OPERATOR_EXECUTED",
        "model":model,
        "revision":revision,
        "remote_structure_ground_digest":remote["ground_digest"],
        "tensor_structure_ground_digest":remote["tensors"]["ground_digest"],
        "operator":"layer0.q_proj[0:4](rms_norm(synthetic_hidden,layer0.input_layernorm.weight))",
        "normalized_input_dimension":hidden,
        "q_projection_output_rows":row_count,
        "actual_remote_weight_bytes_observed":payload,
        "total_remote_weight_payload_bytes":full_payload,
        "fraction_of_weight_payload_observed":round(payload/full_payload,10),
        "manifest_evidence_sha256":_sha(json.dumps(remote,sort_keys=True).encode()),
        "weight_slice_witnesses":[norm_witness,q_witness],
        "input_vector_sha256":_sha(struct.pack(
            "<"+"f"*len(input_vector),*input_vector)),
        "output_coordinates":[round(v,9) for v in computed],
        "reference_max_abs_error_quantized_12dp":round(residual,12),
        "numerical_tolerance_abs":1e-7,
        "independent_numeric_reference_passed":True,
        "full_model_text_generated":False,
        "token_embedding_observed":False,
        "scope":"real_BF16_pretrained_weight_slice_exact_operator_not_complete_LLM",
    }
    receipt["receipt_digest"]=domain_digest(
        "vl64.real-pretrained-model-subgraph-execution.v1",receipt)
    return receipt


def main():
    try:
        result=execute_remote_q_slice()
    except (ValueError,OSError,TimeoutError) as error:
        result={"status":"UNRESOLVED",
                "reason":type(error).__name__,
                "scope":"no_weight_slice_execution_claim_without_exact_remote_bytes"}
    print("VL64_REAL_PRETRAINED_OPERATOR "+json.dumps(
        result,sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
