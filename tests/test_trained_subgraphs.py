"""Numerical subgraph contracts, independent finite scalar oracles and refusals."""
import copy
import math
import random
import pytest

from experiments.model_fabric.trained_subgraphs import (
    centered_rms_norm, standard_rms_norm, dense_matvec, swiglu_expert,
    qwen_gated_delta_layer, deepseek_moe_layer,required_deepseek_expert_tensors,
)
from experiments.model_fabric.architecture_ops import state_digest


def small_qwen_fixture():
    rng=random.Random(819)
    width=3;key_heads=1; value_heads=2;dk=2;dv=2
    specs={'linear_attn.in_proj_qkv.weight':(8,3),
           'linear_attn.in_proj_z.weight':(4,3),
           'linear_attn.in_proj_a.weight':(2,3),
           'linear_attn.in_proj_b.weight':(2,3),
           'linear_attn.out_proj.weight':(3,4),
           'mlp.gate_proj.weight':(4,3),
           'mlp.up_proj.weight':(4,3),
           'mlp.down_proj.weight':(3,4)}
    matrices={name:[[rng.uniform(-.6,.6) for _ in range(cols)] for _ in range(rows)]
              for name,(rows,cols) in specs.items()}
    vectors={'input_layernorm.weight':[.12,-.03,.2],
             'post_attention_layernorm.weight':[.05,.04,-.01],
             'linear_attn.A_log':[-.4,.2],
             'linear_attn.dt_bias':[.6,-.3],
             'linear_attn.norm.weight':[1.05,.95]}
    convolution=[[rng.uniform(-.5,.5) for _ in range(2)] for _ in range(8)]
    def project(name,x):return dense_matvec(matrices[name],x)
    def vector(name):return vectors[name]
    return matrices,vectors,convolution,project,vector


def independent_qwen_step(x,matrices,vectors,conv,prev_conv,prev_recurrent):
    """Standalone scalar reference, not calling VL64 mathematical helpers."""
    def mm(name,x):return [math.fsum(a*b for a,b in zip(row,x)) for row in matrices[name]]
    def normalize(v,g,offset):
        inv=1/math.sqrt(sum(z*z for z in v)/len(v)+1e-6)
        return [z*inv*(w+offset) for z,w in zip(v,g)]
    def sig(z):return 1/(1+math.exp(-z))
    y=normalize(x,vectors['input_layernorm.weight'],1)
    qkv=mm('linear_attn.in_proj_qkv.weight',y)
    z=mm('linear_attn.in_proj_z.weight',y)
    a=mm('linear_attn.in_proj_a.weight',y); b=mm('linear_attn.in_proj_b.weight',y)
    if prev_conv is None:prev_conv=[0.]*8
    c=[]
    for channel in range(8):
        t=conv[channel][0]*prev_conv[channel]+conv[channel][1]*qkv[channel]
        c.append(t/(1+math.exp(-t)))
    # nk=1, nv=2, dk=2, dv=2
    q=c[:2];k=c[2:4];vv=[c[4:6],c[6:8]]
    qn=math.sqrt(sum(t*t for t in q)+1e-6)*math.sqrt(2)
    kn=math.sqrt(sum(t*t for t in k)+1e-6)
    out=[];next_recurrent=[]
    for i in range(2):
        beta=sig(b[i]);log_decay=-math.exp(vectors['linear_attn.A_log'][i])*math.log1p(math.exp(a[i]+vectors['linear_attn.dt_bias'][i]))
        old=[[0.,0.],[0.,0.]] if prev_recurrent is None else prev_recurrent[i]
        scaled=[[v*math.exp(log_decay) for v in row] for row in old]
        pred=[sum(k[j]/kn*scaled[j][col] for j in range(2)) for col in range(2)]
        nxt=[[scaled[j][col]+k[j]/kn*beta*(vv[i][col]-pred[col]) for col in range(2)] for j in range(2)]
        next_recurrent.append(nxt)
        raw=[sum(q[j]/qn*nxt[j][col] for j in range(2)) for col in range(2)]
        scale=1/math.sqrt(sum(t*t for t in raw)/2+1e-6)
        out.extend(raw[col]*scale*vectors['linear_attn.norm.weight'][col]*z[i*2+col]*sig(z[i*2+col]) for col in range(2))
    attn=mm('linear_attn.out_proj.weight',out)
    res=[x[j]+attn[j] for j in range(3)]
    post=normalize(res,vectors['post_attention_layernorm.weight'],1)
    gate=mm('mlp.gate_proj.weight',post);up=mm('mlp.up_proj.weight',post)
    nonlin=[v*sig(v)*u for v,u in zip(gate,up)]
    down=mm('mlp.down_proj.weight',nonlin)
    return [a+b for a,b in zip(res,down)],qkv,next_recurrent


def test_zero_centered_qwen_norm_is_not_gdn_norm():
    x=[-2.,1.,3.]
    assert centered_rms_norm(x,[0.,0.,0.])==pytest.approx([v/(14/3+1e-6)**.5 for v in x])
    assert centered_rms_norm(x,[1.,0.,-1.])==pytest.approx([2*x[0]/(14/3+1e-6)**.5,x[1]/(14/3+1e-6)**.5,0.])
    assert standard_rms_norm(x,[0.,0.,0.])==[0.,0.,0.]
    with pytest.raises(ValueError):centered_rms_norm(x,[1.])


def test_full_qwen_gated_delta_swiglu_layer_three_causal_positions_independent():
    m,v,c,project,vector=small_qwen_fixture()
    conv_state=delta_state=None
    prev_qkv=None;prev_ref=None
    for x in ([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3]):
        out,st,receipt=qwen_gated_delta_layer(x,project,vector,c,key_heads=1,value_heads=2,key_dim=2,value_dim=2,
                                conv_history=conv_state,recurrent_state=delta_state)
        ref,qkv,prev_ref=independent_qwen_step(x,m,v,c,prev_qkv,prev_ref)
        assert out==pytest.approx(ref,abs=1e-12)
        assert len(st['delta'])==2 and all(math.isfinite(a) for mat in st['delta'] for row in mat for a in row)
        assert receipt['state_digest']==state_digest(st['delta'])
        conv_state,delta_state=st['convolution'],st['delta']
        prev_qkv=qkv


def test_qwen_restore_from_frozen_causal_checkpoint_and_rehome_projection():
    _,_,c,p,v=small_qwen_fixture()
    x=[.3,.7,-.2];next_x=[.5,-.8,.2]
    _,st,_=qwen_gated_delta_layer(x,p,v,c,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    saved=copy.deepcopy(st); pin=state_digest(saved['delta'])
    a,n1,_=qwen_gated_delta_layer(next_x,p,v,c,key_heads=1,value_heads=2,key_dim=2,value_dim=2,
                    conv_history=st['convolution'],recurrent_state=st['delta'])
    b,n2,_=qwen_gated_delta_layer(next_x,p,v,c,key_heads=1,value_heads=2,key_dim=2,
                    conv_history=saved['convolution'],recurrent_state=saved['delta'],value_dim=2)
    assert a==b and n1==n2 and state_digest(saved['delta'])==pin


def test_qwen_incomplete_weight_and_causal_error_does_not_change_predecessor():
    _,_,c,p,v=small_qwen_fixture()
    _,st,_=qwen_gated_delta_layer([.1,.2,.3],p,v,c,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    before=copy.deepcopy(st)
    def broken(name,x):
        if name=='mlp.down_proj.weight':raise ValueError('simulated failed host')
        return p(name,x)
    with pytest.raises(ValueError,match='failed host'):
        qwen_gated_delta_layer([.2,.3,.4],broken,v,c,key_heads=1,value_heads=2,key_dim=2,value_dim=2,
                conv_history=st['convolution'],recurrent_state=st['delta'])
    assert st==before
    with pytest.raises(ValueError,match='norm'):
        qwen_gated_delta_layer([.2,.3,.4],p,lambda name:[1.] if name=='linear_attn.norm.weight' else v(name),
                c,key_heads=1,value_heads=2,key_dim=2,value_dim=2)


def test_deepseek_routed_384_expert_ffn_only_six_loaded_independent():
    r=random.Random(55)
    width=4; experts=384; topk=6
    x=[.5,-.1,.3,.2];gamma=[1.2,.95,.85,1.1]
    weights=[[r.uniform(-.3,.3) for _ in range(width)] for _ in range(experts)]
    bias=[r.uniform(-.04,.04) for _ in range(experts)]
    bias[17]+=2.5
    visited=[]
    def callback(i,n):
        visited.append(i)
        # all experts are different and deterministic
        return [math.sin(i*.017)*n[j]*.5 for j in range(width)]
    def shared(n):return [v*.2 for v in n]
    got,receipt=deepseek_moe_layer(x,gamma,weights,bias,callback,shared,topk=topk,route_scale=1.5)
    normalized=[a*b/math.sqrt(sum(z*z for z in x)/width+1e-6) for a,b in zip(x,gamma)]
    logits=[sum(a*b for a,b in zip(row,normalized)) for row in weights]
    scores=[math.sqrt(math.log1p(math.exp(v))) for v in logits]
    ranks=sorted(range(experts),key=lambda i:(-(scores[i]+bias[i]),i))[:topk]
    denom=sum(scores[i] for i in ranks)
    want=[x[j]+normalized[j]*.2+sum(1.5*scores[i]/denom*math.sin(i*.017)*normalized[j]*.5 for i in ranks) for j in range(width)]
    assert got==pytest.approx(want,abs=1e-12)
    assert receipt['selected_experts']==ranks==visited
    assert len(visited)==6 and 17 in visited


def test_deepseek_swiglu_expert_reference():
    x=[.2,.8]
    g=[[1.,.1],[-.3,.5]];u=[[.2,.4],[1.,-.1]];down=[[.5,-.3],[.8,.2]]
    vals=[sum(a*b for a,b in zip(row,x)) for row in g]
    up=[sum(a*b for a,b in zip(row,x)) for row in u]
    n=[v/(1+math.exp(-v))*p for v,p in zip(vals,up)]
    assert swiglu_expert(x,g,u,down)==pytest.approx([sum(a*b for a,b in zip(row,n)) for row in down])


def test_deepseek_selected_expert_range_planner_never_admits_missing_scales():
    names=set()
    for i in (3,13,8):
        for part in ('w1','w3','w2'):
            for suffix in ('weight','scale'):
                names.add(f'layers.5.ffn.experts.{i}.{part}.{suffix}')
    required=required_deepseek_expert_tensors(names,5,[13,8])
    assert [v['expert'] for v in required]==[13,8]
    assert all(f'.experts.3.' not in a for rec in required for p in rec['pairs'] for a in p)
    names.remove('layers.5.ffn.experts.8.w2.scale')
    with pytest.raises(ValueError,match='Incomplete'):
        required_deepseek_expert_tensors(names,5,[13,8])
    with pytest.raises(ValueError):required_deepseek_expert_tensors(names,5,[13,13])
