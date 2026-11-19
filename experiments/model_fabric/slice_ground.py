"""Portable self-contained Ground of *observed* pretrained BF16 tensor slices.

The capsule contains only two exact ranged tensor slices with source hashes
and the fixed typed mathematical plan. Another Python host can reproduce
the same intermediate model operation offline through VL-64's compact core.

Its digest verifies internal integrity, not independent model authenticity;
the capsule cannot produce complete language-model output.
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct

from vld import StructuralLinearOperator,rms_norm,domain_digest


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _decode_bf16(data):
    if len(data)%2:
        raise ValueError("Invalid BF16 byte alignment")
    values=[]
    for (bits,) in struct.iter_unpack("<H",data):
        number=struct.unpack("<f",struct.pack("<I",bits<<16))[0]
        if not math.isfinite(number):
            raise ValueError("Non-finite BF16 model payload")
        values.append(number)
    return values


def compile_slice_ground(*,model,revision,root_ground_digest,
                         tensor_ground_digest,hidden,epsilon,
                         norm_raw,q_raw,source_witnesses):
    if (not isinstance(model,str) or not isinstance(revision,str) or
            not isinstance(root_ground_digest,str) or
            not isinstance(tensor_ground_digest,str) or
            type(hidden) is not int or not 16<=hidden<=2048 or
            not isinstance(epsilon,(int,float)) or not 0<epsilon<=1e-2 or
            not isinstance(norm_raw,bytes) or len(norm_raw)!=hidden*2 or
            not isinstance(q_raw,bytes) or len(q_raw)!=4*hidden*2 or
            not isinstance(source_witnesses,list) or
            len(source_witnesses)!=2):
        raise ValueError("Invalid admitted model slice")
    ground={
        "schema":"vl64.ranged-pretrained-slice-ground.v1",
        "model":model,"revision":revision,
        "remote_ground_digest":root_ground_digest,
        "tensor_header_ground_digest":tensor_ground_digest,
        "hidden_width":hidden,
        "rms_eps":float(epsilon),
        "q_rows":4,
        "norm_bf16_b64":base64.b64encode(norm_raw).decode("ascii"),
        "norm_payload_sha256":_sha(norm_raw),
        "q_bf16_b64":base64.b64encode(q_raw).decode("ascii"),
        "q_payload_sha256":_sha(q_raw),
        "weight_slice_provenance":source_witnesses,
        "operator_path":[
            "rms_norm",
            "first_four_rows_of_pretrained_layer0_q_projection"],
        "training_claim":"actual_pinned_pretrained_model_slice_not_synthetic_weights",
        "result_scope":"model_hidden_vector_subgraph_not_token_generation",
    }
    ground["ground_digest"]=domain_digest(
        "vl64.ranged-pretrained-slice-ground.v1",ground)
    replay_slice_ground(ground)
    return ground


def replay_slice_ground(ground):
    if not isinstance(ground,dict):
        raise ValueError("Ground must be a typed object")
    base={k:v for k,v in ground.items() if k!="ground_digest"}
    if (base.get("schema")!="vl64.ranged-pretrained-slice-ground.v1" or
            domain_digest("vl64.ranged-pretrained-slice-ground.v1",
                          base)!=ground.get("ground_digest")):
        raise ValueError("Remote slice Ground digest mismatch")
    hidden=base.get("hidden_width")
    eps=base.get("rms_eps")
    if (type(hidden) is not int or not 16<=hidden<=2048 or
            type(eps) not in (int,float) or not 0<float(eps)<=1e-2 or
            base.get("q_rows")!=4 or
            base.get("operator_path")!=[
                "rms_norm","first_four_rows_of_pretrained_layer0_q_projection"]):
        raise ValueError("Unadmitted numeric model Ground contract")
    try:
        norm_raw=base64.b64decode(base["norm_bf16_b64"],validate=True)
        q_raw=base64.b64decode(base["q_bf16_b64"],validate=True)
    except (ValueError,TypeError,KeyError) as exc:
        raise ValueError("Malformed encoded tensor weight slices") from exc
    if (len(norm_raw)!=hidden*2 or len(q_raw)!=4*hidden*2 or
            _sha(norm_raw)!=base.get("norm_payload_sha256") or
            _sha(q_raw)!=base.get("q_payload_sha256")):
        raise ValueError("Missing or altered pretrained weight bytes")
    norm=_decode_bf16(norm_raw)
    weights=_decode_bf16(q_raw)
    input_vector=[
        struct.unpack("<f",struct.pack("<f",math.sin((i+1)*0.07)*0.25))[0]
        for i in range(hidden)]
    normalized=rms_norm(input_vector,norm,eps=float(eps))
    op=StructuralLinearOperator(
        rows=4,cols=hidden,
        sparse_delta={(r,c):weights[r*hidden+c]
                      for r in range(4) for c in range(hidden)
                      if weights[r*hidden+c]!=0})
    y=op.matvec(normalized)
    reference_scale=1/math.sqrt(
        sum(value*value for value in input_vector)/hidden+float(eps))
    independent=[
        sum(weights[r*hidden+c]*input_vector[c]*reference_scale*norm[c]
            for c in range(hidden))
        for r in range(4)]
    tolerance=max(abs(a-b) for a,b in zip(y,independent))
    if tolerance>1e-7:
        raise RuntimeError("Locally replayed pretrained tensor differs from reference")
    result={
        "schema":"vl64.replayed-pretrained-slice.v1",
        "status":"VERIFIED_LOCAL_REPLAY_OF_PRETRAINED_TENSOR_SLICE",
        "ground_digest":ground["ground_digest"],
        "actual_model_weight_bytes":len(norm_raw)+len(q_raw),
        "output_coordinates":[round(a,9) for a in y],
        "numeric_check_passed":True,
        "max_abs_error_quantized_12dp":round(tolerance,12),
        "scope":"replay_of_exact_two_tensor_slices_not_full_language_generation",
    }
    result["receipt_digest"]=domain_digest(
        "vl64.replayed-pretrained-slice.v1",result)
    return result
