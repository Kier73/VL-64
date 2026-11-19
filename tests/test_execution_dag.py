"""Independent scalar verification of each pretrained decoder stage and exact-range admission."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import struct
from pathlib import Path

import pytest

from vld import domain_digest
from experiments.model_fabric.execution_dag import (
    compile_decoder_dag, apply_rope, grouped_query_attention,
    swiglu, execute_tokens, MAX_RANGE,
)


def synthetic_pretrained_ground(*,dtype="F32",layers=1,tied=True):
    """A small byte-backed Llama-shaped fixture (not pretrained weights)."""
    r = random.Random(219)
    hidden, vocab, heads, kv, width = 8, 11, 4, 2, 12
    entries = {"model.embed_tokens.weight": [vocab,hidden],
               "model.norm.weight": [hidden]}
    for layer in range(layers):
        p=f"model.layers.{layer}."
        entries.update({p+name: shape for name,shape in {
            "input_layernorm.weight": [hidden],
            "post_attention_layernorm.weight": [hidden],
            "self_attn.q_proj.weight": [hidden,hidden],
            "self_attn.k_proj.weight": [kv*2,hidden],
            "self_attn.v_proj.weight": [kv*2,hidden],
            "self_attn.o_proj.weight": [hidden,hidden],
            "mlp.gate_proj.weight": [width,hidden],
            "mlp.up_proj.weight": [width,hidden],
            "mlp.down_proj.weight": [hidden,width],
        }.items()})
    if not tied:
        entries["lm_head.weight"]=[vocab,hidden]
    weights={}
    binary=bytearray()
    records=[]
    for name, shape in sorted(entries.items()):
        count=math.prod(shape)
        vals = [(0.8+r.random()*0.4) if name.endswith("norm.weight") or "layernorm" in name
                else (r.random()-0.5)*0.3 for _ in range(count)]
        if dtype=="F32":
            part=struct.pack("<"+"f"*count,*vals)
            values=list(struct.unpack("<"+"f"*count,part))
        else:
            part=b"".join(struct.pack("<H",int.from_bytes(struct.pack("<f",x),"little")>>16)
                          for x in vals)
            values=[struct.unpack("<f",struct.pack("<I",v<<16))[0]
                    for (v,) in struct.iter_unpack("<H",part)]
        start=len(binary)
        binary.extend(part)
        records.append({"name":name,"dtype":dtype,"shape":shape,
                        "data_offsets":[start,len(binary)]})
        weights[name] = (shape, values)
    # The offset zero point is Safetensors payload start, after 8-byte header length.
    header_bytes=256
    manifest={"schema":"vl64.remote-tensor-structure-ground.v1",
              "tensor_count":len(records),"tensors":records,
              "declared_payload_bytes":len(binary),
              "header_bytes":header_bytes,
              "source_file_bytes":header_bytes+len(binary)}
    manifest["ground_digest"]=domain_digest(manifest["schema"],manifest)
    ground={"schema":"vl64.remote-model-topology-ground.v1",
            "model":"local/fixture","pinned_revision":"a"*40,
            "tensor_header_status":"OBSERVED_COMPLETE_TENSOR_HEADER",
            "tensors":manifest,
            "architecture_fields":{
                "model_type":"llama","hidden_act":"silu","attention_bias":False,
                "tie_word_embeddings":tied,"hidden_size":hidden,
                "num_hidden_layers":layers,"num_attention_heads":heads,
                "num_key_value_heads":kv,"intermediate_size":width,"vocab_size":vocab,
                "max_position_embeddings":16,"rms_norm_eps":1e-5,"rope_theta":10000.0}}
    ground["ground_digest"]=domain_digest(ground["schema"],ground)
    return ground,bytes(binary),weights


def qualified_host(ground,data,*,observer="local-f32-observer"):
    payload_start=ground["tensors"]["header_bytes"]
    def host(request):
        assert request["byte_end_exclusive"]-request["byte_begin"]<=MAX_RANGE
        a=request["byte_begin"]-payload_start
        b=request["byte_end_exclusive"]-payload_start
        payload=data[a:b]
        return dict(request,payload=payload,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    status="QUALIFIED_EXACT_RANGE",observer_host=observer)
    return host


def scalar_reference(tokens,w,arch):
    """Entire decoder reference without VL64 operators or DAG execution helpers."""
    def m(name, vec):
        shape, values=w[name]
        rows,cols=shape
        return [math.fsum(values[r*cols+c]*vec[c] for c in range(cols)) for r in range(rows)]
    def norm(vec,name):
        weights=w[name][1]
        factor=(math.fsum(z*z for z in vec)/len(vec)+arch["rms_norm_eps"])**-0.5
        return [x*factor*g for x,g in zip(vec,weights)]
    def rotary(x,pos):
        half=len(x)//2
        y=list(x)
        for d in range(half):
            angle=pos*(arch["rope_theta"]**(-d/half))
            c,s=math.cos(angle),math.sin(angle)
            y[d]=x[d]*c-x[d+half]*s
            y[d+half]=x[d+half]*c+x[d]*s
        return y
    cache=[([],[]) for _ in range(arch["num_hidden_layers"])]
    hidden=arch["hidden_size"]
    for pos,token in enumerate(tokens):
        x=w["model.embed_tokens.weight"][1][token*hidden:(token+1)*hidden]
        for layer in range(arch["num_hidden_layers"]):
            p=f"model.layers.{layer}."
            n=norm(x,p+"input_layernorm.weight")
            q=m(p+"self_attn.q_proj.weight",n)
            k=m(p+"self_attn.k_proj.weight",n)
            v=m(p+"self_attn.v_proj.weight",n)
            q=[rotary(q[i*2:(i+1)*2],pos) for i in range(4)]
            k=[rotary(k[i*2:(i+1)*2],pos) for i in range(2)]
            v=[v[i*2:(i+1)*2] for i in range(2)]
            kh,vh=cache[layer]
            kh.append(k)
            vh.append(v)
            context=[]
            for h in range(4):
                keyhead=h//2
                scores=[math.fsum(q[h][d]*frame[keyhead][d] for d in range(2))/math.sqrt(2)
                        for frame in kh]
                max_s=max(scores)
                exps=[math.exp(score-max_s) for score in scores]
                coeff=[e/sum(exps) for e in exps]
                context.extend(math.fsum(c*value[keyhead][d] for c,value in zip(coeff,vh))
                               for d in range(2))
            residual=[a+b for a,b in zip(x,m(p+"self_attn.o_proj.weight",context))]
            post=norm(residual,p+"post_attention_layernorm.weight")
            gate=m(p+"mlp.gate_proj.weight",post)
            up=m(p+"mlp.up_proj.weight",post)
            act=[(g/(1+math.exp(-g)))*u for g,u in zip(gate,up)]
            x=[a+b for a,b in zip(residual,m(p+"mlp.down_proj.weight",act))]
    x=norm(x,"model.norm.weight")
    projection=("model.embed_tokens.weight" if arch["tie_word_embeddings"] else "lm_head.weight")
    return m(projection,x)


def test_pinned_pretrained_manifest_compiles_without_reading_weight_payload():
    path=Path(__file__).resolve().parents[1]/"experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json"
    observed=json.loads(path.read_text())
    compiled=compile_decoder_dag(observed)
    assert compiled["weight_values_observed"] is False
    assert compiled["architecture"]["layers"]==30
    assert compiled["architecture"]["heads"]==9
    assert compiled["architecture"]["kv_heads"]==3
    assert compiled["architecture"]["head_dim"]==64
    assert len(compiled["nodes"])==454
    assert len(compiled["tensors"])==272
    assert compiled["nodes"][-1]["weight"]=="model.embed_tokens.weight"
    assert compiled["dag_digest"]==compile_decoder_dag(observed)["dag_digest"]


@pytest.mark.parametrize("dtype",["F32","BF16"])
@pytest.mark.parametrize("tied",[True,False])
def test_embedding_rope_gqa_swiglu_and_vocab_match_independent_scalar(dtype,tied):
    observed,data,weights=synthetic_pretrained_ground(dtype=dtype,tied=tied)
    dag=compile_decoder_dag(observed)
    tokens=[3,1,7]
    actual=execute_tokens(dag,tokens,qualified_host(observed,data),
                          approved_hosts={"local-f32-observer"})
    expected=scalar_reference(tokens,weights,observed["architecture_fields"])
    assert actual["status"]=="WEIGHT_BACKED_DECODER_FORWARD_EXECUTED"
    assert actual["weight_bytes_acquired"] > 0
    assert actual["exact_range_count"] > 10
    assert len(actual["logits"])==11
    assert max(abs(a-b) for a,b in zip(actual["logits"],expected))<2e-6
    assert actual["selected_next_token"]==max(range(11),key=lambda n:expected[n])
    assert actual["receipt_digest"]==execute_tokens(
        dag,tokens,qualified_host(observed,data),approved_hosts={"local-f32-observer"})["receipt_digest"]


def test_rotary_positions_and_grouped_query_key_sharing():
    assert apply_rope([1.,2.,3.,4.],0,10000.)==[1.,2.,3.,4.]
    x=apply_rope([1.,0.,0.,0.],1,10000.)
    assert x[0]==pytest.approx(math.cos(1))
    assert x[2]==pytest.approx(math.sin(1))
    q=[[1.,0.],[0.,1.],[1.,0.],[0.,1.]]
    k=[[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]
    v=[[[2.,4.],[10.,20.]],[[6.,8.],[30.,40.]]]
    got=grouped_query_attention(q,k,v,heads=4,kv_heads=2)
    assert len(got)==8
    assert got[0] != pytest.approx(got[4])
    assert swiglu([0.,1.],[7.,9.])==pytest.approx([0.,9/(1+math.exp(-1))])


def test_unknown_or_tampered_manifest_fails_closed():
    original, data, _=synthetic_pretrained_ground()
    changed=copy.deepcopy(original)
    changed["architecture_fields"]["hidden_size"]=16
    with pytest.raises(ValueError,match="Ground digest"):
        compile_decoder_dag(changed)
    changed=copy.deepcopy(original)
    changed["architecture_fields"]["hidden_size"]=16
    changed["ground_digest"]=domain_digest(changed["schema"],
                                           {k:v for k,v in changed.items() if k!="ground_digest"})
    with pytest.raises(ValueError,match="tensor"):
        compile_decoder_dag(changed)
    changed=copy.deepcopy(original)
    changed["architecture_fields"]["model_type"]="unsupported"
    changed["ground_digest"]=domain_digest(changed["schema"],
                                           {k:v for k,v in changed.items() if k!="ground_digest"})
    with pytest.raises(ValueError,match="Unimplemented"):
        compile_decoder_dag(changed)


@pytest.mark.parametrize("fault",["bad_hash","wrong_host","short_read","wrong_range","nan"])
def test_weight_admission_rejects_incorrect_or_unqualified_host(fault):
    observed,data,_=synthetic_pretrained_ground()
    dag=compile_decoder_dag(observed)
    host=qualified_host(observed,data)
    def corrupted(request):
        receipt=host(request)
        if fault=="bad_hash":
            receipt["sha256"]="0"*64
        elif fault=="wrong_host":
            receipt["observer_host"]="unknown"
        elif fault=="short_read":
            receipt["payload"]=receipt["payload"][:-1]
        elif fault=="wrong_range":
            receipt["byte_begin"]+=1
        elif fault=="nan":
            receipt["payload"]=struct.pack("<f",float("nan"))+receipt["payload"][4:]
            receipt["sha256"]=hashlib.sha256(receipt["payload"]).hexdigest()
        return receipt
    with pytest.raises(ValueError,match="Unqualified|Nonfinite"):
        execute_tokens(dag,[0],corrupted,approved_hosts={"local-f32-observer"})


def test_core_budget_is_unchanged_by_extension():
    root=Path(__file__).resolve().parents[1]
    assert sum(p.stat().st_size for p in (root/"src/vld").glob("*.py"))<=65536
