"""Pinned, range-qualified Qwen3.8/DeepSeek V4.1 numeric operator probes.

These execute genuine weight *slices* on deterministic synthetic vectors.
Neither probe proves a full pretrained transformer layer or text generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from .architecture_ops import qwen_gates,mx_block_matvec
from .scaled_model_fabric import QualifiedShardNavigator
from .scaled_model_lab import PINNED,_url,_fetch_metadata,pinned_hf_range_host,MAX_INDEX_BYTES

def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _slice(nav,name,*,count):
    d=nav.locate(name)
    if d["dtype"] not in ("BF16","F32"):
        raise ValueError("Unsupported numeric projection dtype")
    raw,_=nav.get_raw(name,flat_start=0,count=count)
    if d["dtype"]=="BF16":
        vals=[struct.unpack("<f",struct.pack("<I",bits<<16))[0] for (bits,) in struct.iter_unpack("<H",raw)]
    else:
        vals=list(struct.unpack("<"+"f"*count,raw))
    if len(vals)!=count or not all(math.isfinite(v) for v in vals):
        raise ValueError("Nonfinite or incomplete observed pretrained tensor")
    return vals,_sha(raw),len(raw)

def probe_qwen_gated_delta_parameters(nav):
    prefix="model.language_model.layers.0.linear_attn."
    names=[prefix+x for x in ("A_log","dt_bias","in_proj_a.weight","in_proj_b.weight")]
    if any(name not in nav.weight_map for name in names):
        raise ValueError("Pinned Qwen gate tensor names are missing")
    dims=[nav.locate(name)["shape"] for name in names]
    if (any(len(dims[j])!=1 or dims[j][0]<1 for j in (0,1)) or
        any(len(dims[j])!=2 or dims[j][0]<1 for j in (2,3)) or
        dims[0]!=dims[1] or dims[0][0]!=dims[2][0] or
        dims[0][0]!=dims[3][0] or dims[2][1]!=dims[3][1]):
        raise ValueError("Qwen layer0 gate shapes differ")
    width=dims[2][1]
    if not 1<=width<=32768:raise ValueError("Unbounded gate projection")
    decoded=[];hashes={};total=0
    for name,count in zip(names,(1,1,width,width)):
        vals,digest,length=_slice(nav,name,count=count)
        decoded.append(vals);hashes[name]=digest;total+=length
    alog,dt,a_row,b_row=decoded
    input_x=[math.sin((j+1)*.013)*.25 for j in range(width)]
    aa=sum(v*x for v,x in zip(a_row,input_x))
    bb=sum(v*x for v,x in zip(b_row,input_x))
    beta,g=qwen_gates([aa],[bb],alog,dt)
    # The reference evaluates decoded bytes with a separate summation order.
    ar=math.fsum(v*x for v,x in zip(a_row,input_x))
    br=math.fsum(v*x for v,x in zip(b_row,input_x))
    beta_ref=1/(1+math.exp(-br)) if br>=0 else math.exp(br)/(1+math.exp(br))
    z=ar+dt[0]
    sp=math.log1p(math.exp(z)) if z<0 else z+math.log1p(math.exp(-z))
    error=max(abs(beta[0]-beta_ref),abs(g[0]+math.exp(alog[0])*sp))
    if not math.isfinite(error) or error>2e-5:
        raise ValueError("Pinned Qwen gate numerical reference differs")
    return {"schema":"vl64.real-large-model-operator.v1",
            "status":"REAL_QWEN_GATED_DELTA_PARAMETERS_VERIFIED",
            "model":nav.index["model"],"revision":nav.index["revision"],
            "tensor_names":names,"tensor_slice_sha256":hashes,
            "numeric_operator":"gated_delta_log_decay_and_beta_head0",
            "input_scope":"synthetic_hidden_vector_not_model_activation",
            "beta":round(beta[0],10),"log_decay":round(g[0],10),
            "max_abs_reference_error":round(error,12),
            "actual_weight_bytes_acquired":total,
            "range_count":nav.range_count,"range_witness_chain_digest":nav.chain,
            "scope":"one real-weight Qwen linear attention gate only; no full layer"}

def probe_deepseek_scaled_fp8_projection(nav):
    name="layers.0.attn.wq_a.weight";scale_name="layers.0.attn.wq_a.scale"
    if name not in nav.weight_map or scale_name not in nav.weight_map:
        raise ValueError("Pinned DeepSeek projection and scale companions missing")
    d=nav.locate(name);s=nav.locate(scale_name)
    if (d["dtype"]!="F8_E4M3" or s["dtype"]!="F8_E8M0" or
        len(d["shape"])!=2 or len(s["shape"])!=2):
        raise ValueError("DeepSeek MXFP8 dtype/shape does not match checkpoint")
    rows,cols=d["shape"]
    if rows<1 or not 32<=cols<=32768 or cols%32 or s["shape"]!=[(rows+31)//32,cols//32]:
        raise ValueError("DeepSeek scale grid incompatible with 32x32 blocks")
    raw,_=nav.get_raw(name,flat_start=0,count=cols)
    scales,_=nav.get_raw(scale_name,flat_start=0,count=cols//32)
    x=[math.cos((j+1)*.031)*.0625 for j in range(cols)]
    got=mx_block_matvec(raw,scales,x,rows=1,cols=cols,format="mxfp8_e4m3",scale_block=(1,32))[0]
    # Independent interpretation of the two exact byte slices.
    reference_terms=[]
    for j,code in enumerate(raw):
        e=(code>>3)&15;m=code&7
        if e==15 and m==7:raise ValueError("Observed nonfinite E4M3 code")
        if scales[j//32]==255:raise ValueError("Observed nonfinite E8M0 scale")
        v=(m/8)*2**-6 if e==0 else (1+m/8)*2**(e-7)
        reference_terms.append((-v if code&0x80 else v)*math.ldexp(1.,scales[j//32]-127)*x[j])
    err=abs(got-math.fsum(reference_terms))
    if not math.isfinite(err) or err>2e-5:
        raise ValueError("Pinned DeepSeek projection numerical reference differs")
    return {"schema":"vl64.real-large-model-operator.v1",
            "status":"REAL_DEEPSEEK_MXFP8_PROJECTION_ROW_VERIFIED",
            "model":nav.index["model"],"revision":nav.index["revision"],
            "tensor_names":[name,scale_name],
            "tensor_slice_sha256":{name:_sha(raw),scale_name:_sha(scales)},
            "numeric_operator":"mxfp8_scaled_matrix_vector_row0",
            "input_scope":"synthetic_hidden_vector_not_model_activation",
            "projection_value":round(got,10),
            "max_abs_reference_error":round(err,12),
            "actual_weight_bytes_acquired":len(raw)+len(scales),
            "range_count":nav.range_count,"range_witness_chain_digest":nav.chain,
            "scope":"one real-weight DeepSeek scaled projection row only; no MoE layer"}

def run(selection):
    model,rev=PINNED[selection]
    index=_fetch_metadata(_url(model,rev,"model.safetensors.index.json"),MAX_INDEX_BYTES)
    nav=QualifiedShardNavigator(model=model,revision=rev,index_raw=index,
        range_host=pinned_hf_range_host(model,rev),
        approved_hosts={"pinned-huggingface-https-range"})
    out=(probe_qwen_gated_delta_parameters(nav) if selection=="qwen"
         else probe_deepseek_scaled_fp8_projection(nav))
    out["index_sha256"]=nav.index["index_sha256"]
    out["index_ground_digest"]=nav.index["ground_digest"]
    return out

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True,choices=("qwen","deepseek"))
    parser.add_argument("--out",default="artifacts/large-model-operator.json")
    args=parser.parse_args(argv)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    try:report=run(args.model)
    except (ValueError,OSError,TimeoutError,OverflowError) as exc:
        report={"status":"UNRESOLVED","model":PINNED[args.model][0],
                "reason":type(exc).__name__,
                "scope":"no pretrained operator verification without exact weights and scales"}
        if isinstance(exc,ValueError):report["refusal"]=str(exc)[:160]
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL64_LARGE_OPERATOR "+json.dumps(report,sort_keys=True))
    return 0 if report["status"].startswith("REAL_") else 1

if __name__=="__main__":
    raise SystemExit(main())
