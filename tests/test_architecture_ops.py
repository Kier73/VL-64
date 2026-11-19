"""Numerical acceptance tests for native Qwen and DeepSeek operator Grounds."""
from __future__ import annotations
import copy
import math
import random
import pytest
from experiments.model_fabric.architecture_ops import (
    qwen_gates,depthwise_causal_conv_step,qwen_gated_delta_step,
    qwen_gated_rms_norm,state_digest,decode_mxfp4,decode_fp8_e4m3fn,
    decode_ue8m0,mx_block_matvec,deepseek_sqrtsoftplus_route,routed_expert_mix,
)

def test_qwen_gate_equations():
    a=[-2.,0.,2.];b=[-2.,0.,2.];logs=[-1.,0.,1.];bias=[.1,-.2,.3]
    beta,g=qwen_gates(a,b,logs,bias)
    for j in range(3):
        assert beta[j]==pytest.approx(math.exp(b[j])/(1+math.exp(b[j])))
        assert g[j]==pytest.approx(-math.exp(logs[j])*math.log(1+math.exp(a[j]+bias[j])))
        assert g[j]<0

def test_causal_conv_kernel_order_and_restart():
    w=[[1.,2.,3.],[-2.,0.,1.]]
    a,h=depthwise_causal_conv_step([2.,5.],w,activation="identity")
    assert a==[6.,5.]
    b,h2=depthwise_causal_conv_step([7.,11.],w,h,activation="identity")
    assert b==[25.,11.] and h2==[[2.,5.],[7.,11.]]
    assert depthwise_causal_conv_step([7.,11.],w,copy.deepcopy(h),activation="identity")[0]==b
    assert h==[[0.,0.],[2.,5.]]

def test_qwen_gated_delta_grouped_state_independent_oracle():
    rng=random.Random(42);state=None
    for _ in range(4):
        dk,dv,nk,nv=3,2,2,4
        q=[[rng.uniform(-2,2) for _ in range(dk)] for _ in range(nk)]
        k=[[rng.uniform(-2,2) for _ in range(dk)] for _ in range(nk)]
        v=[[rng.uniform(-2,2) for _ in range(dv)] for _ in range(nv)]
        beta=[rng.random() for _ in range(nv)];g=[-rng.random() for _ in range(nv)]
        previous=copy.deepcopy(state)
        output,state=qwen_gated_delta_step(q,k,v,g,beta,state)
        for h in range(nv):
            qq=q[h//2];kk=k[h//2]
            qn=math.sqrt(sum(z*z for z in qq)+1e-6)*math.sqrt(dk)
            kn=math.sqrt(sum(z*z for z in kk)+1e-6)
            old=[[0.]*dv for _ in range(dk)] if previous is None else previous[h]
            pred=[sum(kk[j]/kn*old[j][c]*math.exp(g[h]) for j in range(dk)) for c in range(dv)]
            expected=[[old[j][c]*math.exp(g[h])+(kk[j]/kn)*beta[h]*(v[h][c]-pred[c])
                       for c in range(dv)] for j in range(dk)]
            for actual,row in zip(state[h],expected):assert actual==pytest.approx(row,abs=1e-12)
            assert output[h]==pytest.approx([sum(qq[j]/qn*expected[j][c] for j in range(dk))
                                             for c in range(dv)],abs=1e-12)

def test_qwen_gated_rms_norm():
    x=[-1.,2.,3.];gate=[-2.,0.,4.];weight=[1.,2.,3.]
    got=qwen_gated_rms_norm(x,gate,weight)
    inv=1/math.sqrt(sum(z*z for z in x)/3+1e-6)
    assert got==pytest.approx([v*inv*w*g/(1+math.exp(-g)) for v,g,w in zip(x,gate,weight)])
    with pytest.raises(ValueError):qwen_gated_rms_norm(x,gate,[1.])

def test_recovered_state_matches_and_failed_step_does_not_mutate():
    q=[[.2,.5]];k=[[.4,-.2]];v=[[1.1]]
    _,state=qwen_gated_delta_step(q,k,v,[-.4],[.7])
    pin=state_digest(state)
    with pytest.raises(ValueError):qwen_gated_delta_step(q,k,[[float("nan")]],[-.3],[.8],state)
    assert state_digest(state)==pin
    a,s1=qwen_gated_delta_step(q,k,v,[-.3],[.8],state)
    b,s2=qwen_gated_delta_step(q,k,v,[-.3],[.8],copy.deepcopy(state))
    assert a==b and state_digest(s1)==state_digest(s2)

def test_fp4_nibbles():
    assert decode_mxfp4(bytes([0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]),count=16)==[
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]
    with pytest.raises(ValueError):decode_mxfp4(b"\0",count=3)

def test_e4m3fn_known_codes_and_nan():
    assert decode_fp8_e4m3fn(bytes([0,1,8,0x38,0x40,0x7e,0xb8]))==pytest.approx(
        [0.,2**-9,2**-6,1.,2.,448.,-1.])
    with pytest.raises(ValueError):decode_fp8_e4m3fn(b"\x7f")

def test_e8m0_bias_and_nan():
    assert decode_ue8m0(bytes([0,126,127,128,129,254]))==[
        2**-127,.5,1.,2.,4.,2**127]
    with pytest.raises(ValueError):decode_ue8m0(b"\xff")

def test_fp4_block_projection_not_forged_scale():
    raw=bytes([0x22]*16+[0xaa]*16)
    got=mx_block_matvec(raw,bytes([127,128]),[1.]*32,rows=2,cols=32,
                        format="mxfp4_e2m1",scale_block=(1,32))
    assert got==pytest.approx([32.,-64.])
    with pytest.raises(ValueError):mx_block_matvec(raw,b"\x7f",[1.]*32,rows=2,cols=32,
                                    format="mxfp4_e2m1",scale_block=(1,32))

def test_mxfp8_32x32_native_weight_scales():
    raw=bytes([0x38]*(33*32))
    got=mx_block_matvec(raw,bytes([127,128]),[1.]*32,rows=33,cols=32,
                        format="mxfp8_e4m3")
    assert got==pytest.approx([32.]*32+[64.])
    with pytest.raises(ValueError):mx_block_matvec(raw,b"\x7f",[1.]*32,rows=33,cols=32,
                                    format="mxfp8_e4m3")

def test_deepseek_selection_bias_is_not_router_weight():
    logits=[0.,.1,.2,.3];bias=[5.,0.,0.,0.]
    r=deepseek_sqrtsoftplus_route(logits,bias,topk=2,route_scale=1.5)
    assert r["selected_experts"]==[0,3]
    s0=math.sqrt(math.log1p(math.exp(logits[0])))
    s3=math.sqrt(math.log1p(math.exp(logits[3])))
    assert r["routing_weights"]==pytest.approx([1.5*s0/(s0+s3),1.5*s3/(s0+s3)])
    assert deepseek_sqrtsoftplus_route(logits,[0.]*4,topk=2)["selected_experts"]==[3,2]

def test_routed_experts_selected_only():
    r=deepseek_sqrtsoftplus_route([.2,.3,.4],[0.]*3,topk=2)
    called=[]
    def expert(i):
        called.append(i)
        return [float(i+1),2.]
    got=routed_expert_mix(r,expert,[10.,-1.])
    assert called==r["selected_experts"]
    assert got==pytest.approx([10+sum(w*(i+1) for w,i in zip(r["routing_weights"],called)),2.])
    with pytest.raises(ValueError):routed_expert_mix(r,lambda _: [1.],[1.,2.])

def test_refuses_bad_gates_and_unknown_formats():
    with pytest.raises(ValueError):qwen_gated_delta_step([[1.]],[[1.]],[[1.]],[.4],[.5])
    with pytest.raises(ValueError):qwen_gates([1.],[1.],[1.],[])
    with pytest.raises(ValueError):deepseek_sqrtsoftplus_route([1.,2.],[0.],topk=1)
    with pytest.raises(ValueError):mx_block_matvec(b"\0",b"\x7f",[1.],rows=1,cols=1,format="unknown")
