"""Strict synthetic shard-host tests for the two pinned-model numeric probes."""
from __future__ import annotations
import hashlib
import json
import math
import struct
import pytest
from experiments.model_fabric.architecture_probe import (
    probe_qwen_gated_delta_parameters,probe_deepseek_scaled_fp8_projection)
from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator

def bf16(vals):
    return b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",v))[0]>>16) for v in vals)

def nav_for(tensors,*,bad_host=False):
    data=b"";header={}
    for name,(dtype,shape,raw) in sorted(tensors.items()):
        start=len(data);data+=raw
        header[name]={"dtype":dtype,"shape":shape,"data_offsets":[start,len(data)]}
    h=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    blob=struct.pack("<Q",len(h))+h+data
    ix=json.dumps({"weight_map":{name:"weights.safetensors" for name in tensors}},sort_keys=True).encode()
    def observe(req):
        raw=blob[req["byte_begin"]:req["byte_end_exclusive"]]
        return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                "observer_host":"forged-host" if bad_host else "fixture-host",
                "payload":raw,"sha256":hashlib.sha256(raw).hexdigest()}
    return QualifiedShardNavigator(model="example/fixture-model",revision="1"*40,
        index_raw=ix,range_host=observe,approved_hosts={"fixture-host"})

def test_qualified_qwen_four_tensor_gate():
    p="model.language_model.layers.0.linear_attn."
    tensors={
        p+"A_log":("F32",[2],struct.pack("<2f",math.log(.4),math.log(.6))),
        p+"dt_bias":("F32",[2],struct.pack("<2f",.3,.4)),
        p+"in_proj_a.weight":("BF16",[2,32],bf16([.2]*32+[.3]*32)),
        p+"in_proj_b.weight":("BF16",[2,32],bf16([-.1]*32+[.4]*32)),
    }
    nav=nav_for(tensors)
    result=probe_qwen_gated_delta_parameters(nav)
    assert result["status"]=="REAL_QWEN_GATED_DELTA_PARAMETERS_VERIFIED"
    assert result["actual_weight_bytes_acquired"]==136
    assert result["max_abs_reference_error"]<2e-5
    assert result["log_decay"]<0 and 0<result["beta"]<1
    assert nav.weight_bytes==result["actual_weight_bytes_acquired"]

def test_qualified_deepseek_fp8_scale_companion():
    nav=nav_for({"layers.0.attn.wq_a.weight":("F8_E4M3",[32,64],bytes([0x38]*2048)),
                 "layers.0.attn.wq_a.scale":("F8_E8M0",[1,2],bytes([127,128]))})
    result=probe_deepseek_scaled_fp8_projection(nav)
    assert result["status"]=="REAL_DEEPSEEK_MXFP8_PROJECTION_ROW_VERIFIED"
    assert result["actual_weight_bytes_acquired"]==66
    assert result["max_abs_reference_error"]<2e-5
    assert nav.weight_bytes==66

def test_missing_scale_refused_not_replaced_with_one():
    nav=nav_for({"layers.0.attn.wq_a.weight":("F8_E4M3",[32,32],bytes([0x38]*1024))})
    with pytest.raises(ValueError,match="companions"):
        probe_deepseek_scaled_fp8_projection(nav)

def test_wrong_scale_grid_refused():
    nav=nav_for({"layers.0.attn.wq_a.weight":("F8_E4M3",[32,64],bytes([0x38]*2048)),
                 "layers.0.attn.wq_a.scale":("F8_E8M0",[2,1],bytes([127,127]))})
    with pytest.raises(ValueError,match="grid"):
        probe_deepseek_scaled_fp8_projection(nav)

def test_forged_range_host_refused():
    p="model.language_model.layers.0.linear_attn."
    tensors={
        p+"A_log":("F32",[1],struct.pack("<f",-1.)),
        p+"dt_bias":("F32",[1],struct.pack("<f",1.)),
        p+"in_proj_a.weight":("F32",[1,32],struct.pack("<32f",*[1.]*32)),
        p+"in_proj_b.weight":("F32",[1,32],struct.pack("<32f",*[1.]*32)),
    }
    with pytest.raises(ValueError,match="qualified"):
        probe_qwen_gated_delta_parameters(nav_for(tensors,bad_host=True))
