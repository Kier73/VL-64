"""Pretrained model sub-operator executes with *only* selected BF16 weight rows."""
from __future__ import annotations

import math
import struct

import pytest

from experiments.model_fabric.hf_glance import MODEL,REVISION
from experiments.model_fabric import remote_operator_probe as remote


def bf16(values):
    result=bytearray()
    for value in values:
        bits=int.from_bytes(struct.pack("<f",float(value)),"little")
        result+=struct.pack("<H",bits>>16)
    return bytes(result)


def fake_pretrained(monkeypatch):
    hidden=576
    gamma=bf16([1+(i%7)/64 for i in range(hidden)])
    projection=bf16([
        math.sin((r+1)*(c+1)*0.01)*0.1
        for r in range(4) for c in range(hidden)])
    captured=[]
    def fetch(url,*,start=None,stop=None,limit):
        assert start is not None and stop is not None
        assert stop-start+1==limit
        captured.append((start,stop))
        if start==100:
            data=gamma
        elif start==100+hidden*2:
            data=projection
        else:
            raise AssertionError("Full weight download is not authorized")
        return data,{
            "content_range":f"bytes {start}-{stop}/663000",
            "http_status":206,
            "final_hostname":"test.hf.co",
        }
    monkeypatch.setattr(remote,"_read",fetch)
    manifest={
        "ground_digest":"test-exact-manifest",
        "model":MODEL,
        "pinned_revision":REVISION,
        "tensor_header_status":"OBSERVED_COMPLETE_TENSOR_HEADER",
        "architecture_fields":{
            "model_type":"llama",
            "attention_bias":False,
            "hidden_size":hidden,
            "rms_norm_eps":1e-5,
        },
        "tensors":{
            "ground_digest":"observed-header-ground",
            "header_bytes":100,
            "tensor_count":2,
            "declared_payload_bytes":663000,
            "tensors":[
                {"name":"model.layers.0.input_layernorm.weight",
                 "dtype":"BF16","shape":[hidden],
                 "data_offsets":[0,hidden*2]},
                {"name":"model.layers.0.self_attn.q_proj.weight",
                 "dtype":"BF16","shape":[hidden,hidden],
                 "data_offsets":[hidden*2,hidden*2+hidden*hidden*2]},
            ],
        },
    }
    return manifest,captured,gamma,projection


def test_selected_pretrained_weight_ranges_execute_through_VL64(monkeypatch):
    manifest,captured,gamma,projection=fake_pretrained(monkeypatch)
    result=remote.execute_remote_q_slice(observed=manifest)
    assert result["status"]=="ACTUAL_REMOTE_PRETRAINED_WEIGHT_OPERATOR_EXECUTED"
    assert result["normalized_input_dimension"]==576
    assert result["q_projection_output_rows"]==4
    assert result["actual_remote_weight_bytes_observed"]==5760
    assert result["reference_max_abs_error_quantized_12dp"]<=1e-7
    assert result["independent_numeric_reference_passed"] is True
    assert result["full_model_text_generated"] is False
    assert result["token_embedding_observed"] is False
    assert captured==[(100,1251),(1252,5859)]
    assert len(result["weight_slice_witnesses"])==2
    assert all(w["http_status"]==206 for w in result["weight_slice_witnesses"])
    assert len(result["output_coordinates"])==4


def test_selected_weight_operator_refuses_unobserved_or_mismatched_model(monkeypatch):
    manifest,captured,_,_=fake_pretrained(monkeypatch)
    bad=dict(manifest)
    bad["tensor_header_status"]="UNRESOLVED"
    with pytest.raises(ValueError,match="header"):
        remote.execute_remote_q_slice(observed=bad)
    bad=dict(manifest)
    bad["architecture_fields"]=dict(manifest["architecture_fields"])
    bad["architecture_fields"]["model_type"]="unknown"
    with pytest.raises(ValueError,match="family"):
        remote.execute_remote_q_slice(observed=bad)
    assert captured==[]


def test_selected_weight_operator_rejects_unsupported_bf16_values(monkeypatch):
    manifest,captured,_,_=fake_pretrained(monkeypatch)
    old=remote._read
    def with_nan(*args,**kwargs):
        payload,receipt=old(*args,**kwargs)
        if kwargs["start"]==100:
            payload=b"\xc0\x7f"+payload[2:]  # BF16 quiet NaN
        return payload,receipt
    monkeypatch.setattr(remote,"_read",with_nan)
    with pytest.raises(ValueError,match="Nonfinite"):
        remote.execute_remote_q_slice(observed=manifest)


def test_remote_range_parser_requires_pinned_huggingface_model_and_revision():
    from experiments.model_fabric.hf_glance import _url
    assert _url(MODEL,REVISION,"config.json").startswith("https://huggingface.co/")
    with pytest.raises(ValueError):
        _url("../../evil",REVISION,"config.json")
    with pytest.raises(ValueError):
        _url(MODEL,"main","model.safetensors")
    with pytest.raises(ValueError):
        _url(MODEL,REVISION,"model.pickle")
