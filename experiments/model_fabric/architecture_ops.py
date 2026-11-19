"""Finite, scale-independent Qwen Gated DeltaNet and DeepSeek MX/MoE primitives.

Model bytes and trust are admitted separately by QualifiedShardNavigator.
This adapter changes nothing in the 64-KiB VL-64 core.
"""
from __future__ import annotations
import hashlib
import json
import math

def _finite(values,label):
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError(label+" contains nonfinite or untyped values")

def state_digest(value):
    def check(x):
        if isinstance(x,(list,tuple)): return [check(a) for a in x]
        if type(x) in (float,int) and math.isfinite(x): return float(x)
        raise ValueError("Invalid causal state")
    raw=json.dumps(check(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(b"vl64.architecture-state.v1\0"+raw).hexdigest()

def _softplus(x):
    return max(x,0.0)+math.log1p(math.exp(-abs(x)))

def _sigmoid(x):
    if x>=0:return 1./(1.+math.exp(-x))
    e=math.exp(x);return e/(1.+e)

def qwen_gates(a,b,a_log,dt_bias):
    if len({len(a),len(b),len(a_log),len(dt_bias)})!=1 or not a:
        raise ValueError("Qwen gate dimensions differ")
    for vals in (a,b,a_log,dt_bias):_finite(vals,"Qwen gate")
    beta=[_sigmoid(x) for x in b]
    decay=[-math.exp(z)*_softplus(x+y) for z,x,y in zip(a_log,a,dt_bias)]
    if any(not math.isfinite(z) or z>0 for z in decay):
        raise ValueError("Unqualified Qwen gate decay")
    return beta,decay

def depthwise_causal_conv_step(x,weights,history=None,*,activation="silu"):
    width=len(x)
    if not width or len(weights)!=width:raise ValueError("Invalid convolution width")
    kernel=len(weights[0])
    if kernel<1 or kernel>64 or any(len(w)!=kernel for w in weights):
        raise ValueError("Unqualified causal convolution kernel")
    history=[] if history is None else history
    if len(history)>kernel-1 or any(len(z)!=width for z in history):
        raise ValueError("Unqualified convolution past")
    _finite(x,"conv input")
    for row in list(weights)+list(history):_finite(row,"conv state")
    padded=[[0.]*width for _ in range(max(0,kernel-1-len(history)))]+[list(z) for z in history]+[list(x)]
    result=[]
    for c in range(width):
        z=math.fsum(weights[c][j]*padded[j][c] for j in range(kernel))
        if activation=="silu":z*=_sigmoid(z)
        elif activation!="identity":raise ValueError("Unsupported convolution activation")
        result.append(z)
    _finite(result,"conv output")
    return result,padded[-(kernel-1):] if kernel>1 else []

def _l2(x):
    norm=math.sqrt(math.fsum(z*z for z in x)+1e-6)
    return [v/norm for v in x]

def qwen_gated_delta_step(query,key,value,log_decay,beta,state=None):
    nk=len(key);nv=len(value)
    if not nk or not nv or len(query)!=nk or nv%nk or len(beta)!=nv or len(log_decay)!=nv:
        raise ValueError("Invalid gated delta head count")
    dk=len(key[0]);dv=len(value[0])
    if dk<1 or dv<1 or any(len(v)!=dk for v in key+query) or any(len(v)!=dv for v in value):
        raise ValueError("Invalid gated delta dimensions")
    for row in key+query+value:_finite(row,"gated delta input")
    _finite(beta,"beta");_finite(log_decay,"decay")
    if any(not 0<=x<=1 for x in beta) or any(x>0 for x in log_decay):
        raise ValueError("Invalid gated-delta gate")
    if state is None:state=[[[0.]*dv for _ in range(dk)] for _ in range(nv)]
    if len(state)!=nv or any(len(m)!=dk or any(len(row)!=dv for row in m) for m in state):
        raise ValueError("Gated delta state has incorrect shape")
    for matrix in state:
        for row in matrix:_finite(row,"gated delta causal state")
    q=[_l2(x) for x in query];k=[_l2(x) for x in key]
    output=[];next_state=[]
    for h in range(nv):
        idx=h//(nv//nk)
        kh=k[idx];qh=[z/math.sqrt(dk) for z in q[idx]]
        faded=[[z*math.exp(log_decay[h]) for z in row] for row in state[h]]
        prediction=[math.fsum(kh[j]*faded[j][c] for j in range(dk)) for c in range(dv)]
        delta=[beta[h]*(value[h][c]-prediction[c]) for c in range(dv)]
        updated=[[faded[j][c]+kh[j]*delta[c] for c in range(dv)] for j in range(dk)]
        output.append([math.fsum(qh[j]*updated[j][c] for j in range(dk)) for c in range(dv)])
        next_state.append(updated)
    for row in output:_finite(row,"gated delta output")
    return output,next_state

def qwen_gated_rms_norm(x,gate,weight,eps=1e-6):
    if not x or len(x)!=len(gate) or len(x)!=len(weight) or eps<=0:
        raise ValueError("Qwen gated norm dimensions differ")
    for v in (x,gate,weight):_finite(v,"gated norm")
    inv=1/math.sqrt(math.fsum(z*z for z in x)/len(x)+eps)
    return [v*inv*w*(g*_sigmoid(g)) for v,w,g in zip(x,weight,gate)]

_E2M1=(0.,.5,1.,1.5,2.,3.,4.,6.)
def decode_mxfp4(raw,*,count):
    if type(count) is not int or count<=0 or len(raw)!=(count+1)//2:
        raise ValueError("Unqualified packed MXFP4 extent")
    out=[]
    for i in range(count):
        nib=(raw[i//2]>>(4*(i%2)))&15
        out.append(_E2M1[nib&7]*(-1 if nib&8 else 1))
    return out

def decode_fp8_e4m3fn(raw):
    if not isinstance(raw,bytes) or not raw:raise ValueError("Unqualified FP8 bytes")
    out=[]
    for code in raw:
        sign=-1. if code&128 else 1.
        e=(code>>3)&15;m=code&7
        if e==15 and m==7:raise ValueError("Nonfinite E4M3 code")
        out.append(sign*(m*2.**-9 if e==0 else (1+m/8)*2.**(e-7)))
    return out

def decode_ue8m0(raw):
    if not isinstance(raw,bytes) or not raw or 255 in raw:
        raise ValueError("Invalid E8M0 exponent-only scale")
    return [math.ldexp(1.,int(b)-127) for b in raw]

def mx_block_matvec(weight_raw,scale_raw,x,*,rows,cols,format,scale_block=(32,32)):
    if (type(rows) is not int or type(cols) is not int or not 1<=rows<=65536 or
        not 1<=cols<=65536 or len(x)!=cols or type(scale_block) is not tuple or
        len(scale_block)!=2 or any(type(v) is not int or v<=0 for v in scale_block)):
        raise ValueError("Invalid MX tensor shape")
    _finite(x,"MX activation")
    if format=="mxfp8_e4m3":
        vals=decode_fp8_e4m3fn(weight_raw)
        if len(vals)!=rows*cols:raise ValueError("Incorrect FP8 payload shape")
        br,bc=scale_block
    elif format=="mxfp4_e2m1":
        if cols%32 or scale_block!=(1,32):
            raise ValueError("MXFP4 requires per-row K=32 E8M0 scales")
        vals=decode_mxfp4(weight_raw,count=rows*cols);br,bc=scale_block
    else:raise ValueError("Unsupported MX format")
    nc=(cols+bc-1)//bc;scales=decode_ue8m0(scale_raw)
    if len(scales)!=((rows+br-1)//br)*nc:
        raise ValueError("MX block scale shape does not match weights")
    return [math.fsum(vals[r*cols+c]*scales[(r//br)*nc+c//bc]*x[c]
                      for c in range(cols)) for r in range(rows)]

def deepseek_sqrtsoftplus_route(logits,selection_bias,*,topk=6,route_scale=1.5):
    n=len(logits)
    if not n or len(selection_bias)!=n or type(topk) is not int or not 1<=topk<=n or not 0<route_scale<=100:
        raise ValueError("Invalid expert routing dimensions")
    _finite(logits,"router logits");_finite(selection_bias,"router selection bias")
    scores=[math.sqrt(_softplus(x)) for x in logits]
    selected=sorted(range(n),key=lambda i:(-(scores[i]+selection_bias[i]),i))[:topk]
    denom=math.fsum(scores[i] for i in selected)
    if denom<=0 or not math.isfinite(denom):
        raise ValueError("Invalid expert probability normalization")
    return {"selected_experts":selected,
            "routing_weights":[route_scale*scores[i]/denom for i in selected],
            "selection_only_bias":True,"status":"NUMERICAL_ROUTER_EXECUTED"}

def routed_expert_mix(router,expert_callable,shared_output):
    if router.get("status")!="NUMERICAL_ROUTER_EXECUTED" or not callable(expert_callable):
        raise ValueError("Unadmitted MoE route")
    ids=router["selected_experts"];weights=router["routing_weights"]
    if len(ids)!=len(weights) or len(set(ids))!=len(ids) or not shared_output:
        raise ValueError("Inconsistent routed expert receipt")
    _finite(shared_output,"shared expert")
    parts=[]
    for expert in ids:
        x=expert_callable(expert)
        if len(x)!=len(shared_output):raise ValueError("Expert output shape differs")
        _finite(x,"expert output");parts.append(x)
    return [shared_output[j]+math.fsum(w*v[j] for w,v in zip(weights,parts))
            for j in range(len(shared_output))]
