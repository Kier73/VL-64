"""Optional pinned pretrained SmolLM2 layer-0 experiment with scalar reference.

Reads just the selected token rows and nine layer-0 BF16 tensors in bounded
HTTP ranges. A distinct pure-Python scalar implementation consumes exactly
the observed bytes and checks the VL64 numerical result; it never treats
missing weights or a failed remote request as verified execution.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

from .execution_dag import compile_decoder_dag,huggingface_range_host
from .pretrained_layer import execute_pretrained_layer,LAYER_TENSORS

MANIFEST=Path(__file__).parent/"datasets/2026-11-18-pinned-smollm2-tensor-ground.json"
TOKENS=(42,75)


def _decode(data,dtype):
    if dtype=="BF16":
        return [struct.unpack("<f",struct.pack("<I",x<<16))[0]
                for (x,) in struct.iter_unpack("<H",data)]
    if dtype=="F32":
        return list(struct.unpack("<"+"f"*(len(data)//4),data))
    raise ValueError("Unrecognized observer tensor dtype")


def independent_reference(plan,captured,tokens):
    """Dense scalar reference without any VL64 mathematical primitives."""
    a=plan["architecture"]
    h,d,nh,nkv=a["hidden"],a["head_dim"],a["heads"],a["kv_heads"]
    def get(name,start=0,count=None):
        tensor=plan["tensors"][name]
        dtype=tensor["dtype"]
        size=2 if dtype=="BF16" else 4
        if count is None:
            count=math.prod(tensor["shape"])
        begin=plan["header_bytes"]+tensor["data_offsets"][0]+start*size
        end=begin+count*size
        fragments=sorted(captured[name],key=lambda part:part[0])
        raw=bytearray()
        cursor=begin
        for first,data in fragments:
            last=first+len(data)
            if last<=cursor:
                continue
            if first>cursor:
                break
            raw.extend(data[cursor-first:min(last,end)-first])
            cursor=min(last,end)
            if cursor==end:
                break
        if cursor!=end:
            raise ValueError("Independent decoder could not reconstruct admitted ranges")
        return _decode(bytes(raw),dtype)
    weights={suffix:get("model.layers.0."+suffix) for suffix in LAYER_TENSORS}
    def linear(suffix,x):
        name="model.layers.0."+suffix
        rows,cols=plan["tensors"][name]["shape"]
        w=weights[suffix]
        return [math.fsum(w[r*cols+j]*x[j] for j in range(cols))
                for r in range(rows)]
    def norm(x,suffix):
        gamma=weights[suffix]
        scale=1/math.sqrt(math.fsum(z*z for z in x)/len(x)+a["eps"])
        return [z*scale*g for z,g in zip(x,gamma)]
    def rotary(x,pos):
        half=len(x)//2
        y=x.copy()
        for j in range(half):
            angle=pos/(a["theta"]**(j/half))
            co,si=math.cos(angle),math.sin(angle)
            y[j]=x[j]*co-x[j+half]*si
            y[j+half]=x[j]*si+x[j+half]*co
        return y
    def silu(z):
        return z/(1+math.exp(-z)) if z>=0 else z*math.exp(z)/(1+math.exp(z))
    keys,values=[],[]
    for pos,token in enumerate(tokens):
        x=get("model.embed_tokens.weight",token*h,h)
        n=norm(x,"input_layernorm.weight")
        q=linear("self_attn.q_proj.weight",n)
        k=linear("self_attn.k_proj.weight",n)
        v=linear("self_attn.v_proj.weight",n)
        qh=[rotary(q[j*d:(j+1)*d],pos) for j in range(nh)]
        kh=[rotary(k[j*d:(j+1)*d],pos) for j in range(nkv)]
        vh=[v[j*d:(j+1)*d] for j in range(nkv)]
        keys.append(kh)
        values.append(vh)
        context=[]
        for head in range(nh):
            shared=head//(nh//nkv)
            scores=[math.fsum(qh[head][j]*frame[shared][j] for j in range(d))/math.sqrt(d)
                    for frame in keys]
            peak=max(scores)
            exps=[math.exp(s-peak) for s in scores]
            probabilities=[v/math.fsum(exps) for v in exps]
            context.extend(math.fsum(probabilities[t]*values[t][shared][j]
                                     for t in range(len(values))) for j in range(d))
        attn=linear("self_attn.o_proj.weight",context)
        residual=[z+w for z,w in zip(x,attn)]
        post=norm(residual,"post_attention_layernorm.weight")
        gate=linear("mlp.gate_proj.weight",post)
        up=linear("mlp.up_proj.weight",post)
        activation=[silu(g)*u for g,u in zip(gate,up)]
        down=linear("mlp.down_proj.weight",activation)
        result=[z+w for z,w in zip(residual,down)]
    return result


def run(*,tokens=TOKENS):
    ground=json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan=compile_decoder_dag(ground)
    remote=huggingface_range_host(plan)
    captured=defaultdict(list)
    def host(request):
        response=remote(request)
        captured[request["tensor"]].append((request["byte_begin"],response["payload"]))
        return response
    result=execute_pretrained_layer(
        plan,list(tokens),host,approved_hosts={"pinned-huggingface-https-range"})
    reference=independent_reference(plan,captured,tokens)
    got=result["output"]
    if len(got)!=len(reference):
        raise ValueError("Pretrained reference output shape differs")
    error=max(abs(x-y) for x,y in zip(got,reference))
    tol=2e-5
    if not math.isfinite(error) or error>tol:
        raise ValueError("Pretrained layer disagrees with independent scalar reference")
    return {"schema":"vl64.real-pretrained-layer-check.v1",
            "status":"REAL_BF16_LAYER0_INDEPENDENTLY_VERIFIED",
            "model":plan["model"],"revision":plan["revision"],
            "tokens":list(tokens),"layer":0,
            "dag_digest":plan["dag_digest"],
            "layer_execution_receipt_digest":result["receipt_digest"],
            "last_hidden_first_8":[round(x,8) for x in got[:8]],
            "independent_reference_max_abs_error":round(error,12),
            "reference_tolerance_abs":tol,
            "weight_payload_bytes_acquired":result["weight_payload_bytes_acquired"],
            "exact_range_count":result["exact_range_count"],
            "range_witness_chain_digest":result["range_witness_chain_digest"],
            "scope":"one_real_pretrained_layer_only_not_complete_LLM_generation"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts/pretrained-layer0-report.json")
    args=parser.parse_args()
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run()
    except (ValueError,OSError,TimeoutError) as error:
        result={"status":"UNRESOLVED","reason":type(error).__name__,
                "scope":"no_pretrained_layer_verification_without_exact_weights"}
        target.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print("VL64_REAL_PRETRAINED_LAYER "+json.dumps(result,sort_keys=True))
        raise SystemExit(1) from error
    target.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("VL64_REAL_PRETRAINED_LAYER "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
