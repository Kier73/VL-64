"""Optional LIVE 30-layer pinned SmolLM2 prefill, process restart, and logits.

A pure VL-64 implementation executes the typed DAG; a separately written
NumPy reference verifies each REAL pretrained layer and the vocabulary logits.
All weights come from approved, pinned, bounded 206-Range responses; an
unavailable HTTP source yields UNRESOLVED and a nonzero process exit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .execution_dag import compile_decoder_dag, huggingface_range_host
from .full_decoder import StreamingDecoder,save_full_checkpoint,load_full_checkpoint
from .pretrained_orbit import ContentAddressedPages

MANIFEST = Path(__file__).parent / "datasets/2026-11-18-pinned-smollm2-tensor-ground.json"


def _reference_tensor(plan, captured, name):
    """Independently reconstruct the exact source bytes from range responses."""
    import numpy as np
    tensor=plan["tensors"][name]
    size=2 if tensor["dtype"]=="BF16" else 4
    start=plan["header_bytes"]+tensor["data_offsets"][0]
    end=start+math.prod(tensor["shape"])*size
    cursor=start
    raw=bytearray()
    for offset,part in sorted(captured[name]):
        if offset!=cursor or offset+len(part)>end:
            raise ValueError("Reference did not independently reconstruct exact tensor extent")
        raw.extend(part)
        cursor+=len(part)
    if cursor!=end:
        raise ValueError("Independent reference missing pretrained weight ranges")
    if tensor["dtype"]=="BF16":
        words=np.frombuffer(raw,dtype="<u2").astype("<u4")
        values=(words<<16).view("<f4").astype(np.float64)
    else:
        values=np.frombuffer(raw,dtype="<f4").astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite independent pretrained reference weight")
    return values.reshape(tensor["shape"])


def independent_numpy_layer(plan, captured, index, incoming):
    """Dense NumPy linear algebra, not the VL64 StructuralLinearOperator path."""
    import numpy as np
    a=plan["architecture"]
    h=a["hidden"]
    heads,kvheads,d=a["heads"],a["kv_heads"],a["head_dim"]
    p=f"model.layers.{index}."
    names=("input_layernorm.weight","self_attn.q_proj.weight",
           "self_attn.k_proj.weight","self_attn.v_proj.weight",
           "self_attn.o_proj.weight","post_attention_layernorm.weight",
           "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight")
    w={name:_reference_tensor(plan,captured,p+name) for name in names}
    def norm(vec,name):
        return vec / np.sqrt(np.mean(vec*vec)+a["eps"])*w[name]
    keys=[]
    values=[]
    outputs=[]
    angles_den=a["theta"]**(2*np.arange(d//2,dtype=np.float64)/d)
    def rope(q,pos):
        angles=pos/angles_den
        co=np.cos(angles)
        si=np.sin(angles)
        rotated=np.empty_like(q)
        rotated[:,:d//2]=q[:,:d//2]*co-q[:,d//2:]*si
        rotated[:,d//2:]=q[:,:d//2]*si+q[:,d//2:]*co
        return rotated
    for pos,hidden in enumerate(incoming):
        x=np.asarray(hidden,dtype=np.float64)
        if x.shape!=(h,):
            raise ValueError("Reference hidden-state width mismatch")
        z=norm(x,"input_layernorm.weight")
        q=rope((w["self_attn.q_proj.weight"]@z).reshape(heads,d),pos)
        k=rope((w["self_attn.k_proj.weight"]@z).reshape(kvheads,d),pos)
        v=(w["self_attn.v_proj.weight"]@z).reshape(kvheads,d)
        keys.append(k)
        values.append(v)
        history_k=np.stack(keys)
        history_v=np.stack(values)
        context=[]
        for head in range(heads):
            shared=head//(heads//kvheads)
            scores=history_k[:,shared,:]@q[head] / math.sqrt(d)
            exps=np.exp(scores-np.max(scores))
            probs=exps/np.sum(exps)
            context.extend(probs@history_v[:,shared,:])
        aout=w["self_attn.o_proj.weight"]@np.asarray(context)
        residual=x+aout
        z=norm(residual,"post_attention_layernorm.weight")
        gate=w["mlp.gate_proj.weight"]@z
        up=w["mlp.up_proj.weight"]@z
        # Numerically stable sigmoid, matching the VL-64 scalar SiLU path.
        sigmoid=np.empty_like(gate)
        pos_mask=gate>=0
        sigmoid[pos_mask]=1/(1+np.exp(-gate[pos_mask]))
        exp_neg=np.exp(gate[~pos_mask])
        sigmoid[~pos_mask]=exp_neg/(1+exp_neg)
        down=w["mlp.down_proj.weight"]@((gate*sigmoid)*up)
        out=residual+down
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite reference pretrained layer output")
        outputs.append(out)
    return outputs


def independent_numpy_vocab(plan,captured,hidden):
    import numpy as np
    a=plan["architecture"]
    gamma=_reference_tensor(plan,captured,"model.norm.weight")
    x=np.asarray(hidden,dtype=np.float64)
    x=x/np.sqrt(np.mean(x*x)+a["eps"])*gamma
    proj="model.embed_tokens.weight" if a["tied_embedding"] else "lm_head.weight"
    return _reference_tensor(plan,captured,proj)@x


def run(*,tokens=(42,75),through_layer=None,state_dir="artifacts/full-pretrained",
        resume_from=None,expected_head=None,owner="first-process",epoch=0,
        result_path="artifacts/full-pretrained-30-report.json"):
    import numpy as np
    observed=json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan=compile_decoder_dag(observed)
    if plan["architecture"]["layers"]!=30:
        raise ValueError("Pinned-model experiment requires observed 30-layer DAG")
    count=plan["architecture"]["layers"]
    end=count if through_layer is None else through_layer
    if type(end) is not int or not 1<=end<=count:
        raise ValueError("Invalid requested pretrained layer frontier")
    store=ContentAddressedPages(Path(state_dir)/"pages")
    remote=huggingface_range_host(plan)
    captured=defaultdict(list)
    def host(request):
        reply=remote(request)
        captured[request["tensor"]].append((request["byte_begin"],reply["payload"]))
        return reply
    resume=load_full_checkpoint(resume_from) if resume_from else None
    session=StreamingDecoder(plan,list(tokens),lambda i:host,
        lambda i:{"pinned-huggingface-https-range"},store,
        owner=owner,epoch=epoch,checkpoint=resume,expected_head=expected_head)
    layer_checks=[]
    output=Path(result_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_path=Path(state_dir)/"checkpoint.json"
    for index in range(session.completed_layers,end):
        prior=session.hidden
        captured.clear()
        progress=session.advance_layer()
        # Real pretrained independent arithmetic, including every coordinate.
        independent=independent_numpy_layer(plan,captured,index,prior)
        error=max(float(np.max(np.abs(np.asarray(ours)-reference)))
                  for ours,reference in zip(session.hidden,independent))
        tolerance=2e-4
        if not math.isfinite(error) or error>tolerance:
            raise ValueError("Real pretrained layer differs from independent NumPy reference")
        layer_checks.append({"index":index,"max_abs_error":round(error,10),
                             "tolerance_abs":tolerance,
                             "hidden_sha256":progress["hidden_sha256"],
                             "static_weight_identity":progress["static_weight_identity"],
                             "bytes_acquired":progress["layer_weight_bytes"],
                             "range_reads":progress["layer_range_reads"]})
        snap=session.checkpoint()
        save_full_checkpoint(checkpoint_path,snap)
        print("VL64_PRETRAINED_LAYER "+json.dumps({"layer":index,
            "completed":session.completed_layers,"total":count,"error":error,
            "checkpoint_digest":snap["checkpoint_digest"]},sort_keys=True),flush=True)
    snap=session.checkpoint()
    head=save_full_checkpoint(checkpoint_path,snap)
    result={"schema":"vl64.real-pretrained-30-layer-run.v1",
            "model":plan["model"],"revision":plan["revision"],
            "dag_digest":plan["dag_digest"],
            "tokens":list(tokens),"layers_completed":session.completed_layers,
            "total_declared_layers":count,"checkpoint_digest":head,
            "previous_checkpoint_digest":session.previous_checkpoint,
            "range_reads_cumulative":session.ranges_acquired,
            "weight_payload_bytes_cumulative":session.bytes_acquired,
            "verified_this_process":layer_checks,
            "verified_layers_this_process":len(layer_checks),
            "claim":"all layer operands individually range-qualified; independent numeric check per completed layer"}
    if end==count:
        captured.clear()
        receipt,logits=session.logits(return_logits=True)
        independently=independent_numpy_vocab(plan,captured,session.hidden[-1])
        error=float(np.max(np.abs(np.asarray(logits)-independently)))
        if not math.isfinite(error) or error>2e-4:
            raise ValueError("Pretrained full vocabulary differs from independent reference")
        if int(np.argmax(independently))!=receipt["selected_next_token"]:
            raise ValueError("Pretrained reference top-token disagreement")
        result.update({"status":"REAL_PRETRAINED_30_LAYER_LOGITS_INDEPENDENTLY_VERIFIED",
                       "selected_next_token":receipt["selected_next_token"],
                       "top_token_ids":receipt["top_token_ids"],
                       "top_logits":receipt["top_logits"],
                       "full_vocab_size":receipt["full_vocab_size"],
                       "logits_f64_sha256":receipt["logits_f64_sha256"],
                       "independent_logit_max_abs_error":round(error,10),
                       "result_receipt_digest":receipt["receipt_digest"],
                       "range_reads_including_logits":receipt["exact_range_count"],
                       "weight_bytes_including_logits":receipt["weight_payload_bytes_acquired"],
                       "scope":"real_supplied_token_ids_full_30_layer_forward_not_tokenization_or_remote_processor_execution"})
    else:
        result.update({"status":"VERIFIED_PRETRAINED_LAYER_FRONTIER",
                       "scope":"partial_pretrained_layer_run_no_final_logits"})
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL64_CHECKPOINT_HEAD "+head,flush=True)
    print("VL64_PRETRAINED_30 "+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,nargs="+",default=[42,75])
    p.add_argument("--through-layer",type=int,default=30)
    p.add_argument("--state-dir",default="artifacts/full-pretrained")
    p.add_argument("--resume-from")
    p.add_argument("--expected-head")
    p.add_argument("--owner",default="first-process")
    p.add_argument("--epoch",type=int,default=0)
    p.add_argument("--result-out",default="artifacts/full-pretrained-30-report.json")
    p.add_argument("--github-output",help="Append checkpoint_head to GitHub step output")
    args=p.parse_args()
    try:
        result=run(tokens=args.tokens,through_layer=args.through_layer,
                   state_dir=args.state_dir,resume_from=args.resume_from,
                   expected_head=args.expected_head,owner=args.owner,
                   epoch=args.epoch,result_path=args.result_out)
        if args.github_output:
            with Path(args.github_output).open("a",encoding="utf-8") as output:
                output.write("checkpoint_head="+result["checkpoint_digest"]+"\n")
    except (ValueError,OSError,TimeoutError) as exc:
        # Fail instead of manufacturing a verified pretrained model result.
        target=Path(args.result_out)
        target.parent.mkdir(parents=True,exist_ok=True)
        status={"status":"UNRESOLVED","error_type":type(exc).__name__,
                "reason":str(exc)[:300],"tokens":args.tokens,
                "target_layers":args.through_layer,
                "scope":"no verified layer/frontier claim if pretrained range or numeric check fails"}
        target.write_text(json.dumps(status,sort_keys=True,indent=2)+"\n")
        print("VL64_PRETRAINED_30 "+json.dumps(status,sort_keys=True),flush=True)
        raise SystemExit(1) from exc


if __name__=="__main__":
    main()
