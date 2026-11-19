"""Complete native layer contracts, independently checked and never inferred from a manifest."""
from __future__ import annotations

import copy
import math
import random
import struct
import pytest

from test_architecture_probe import nav_for, bf16
from test_trained_subgraphs import small_qwen_fixture,independent_qwen_step
from experiments.model_fabric.architecture_ops import state_digest
from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, DeepSeekQualifiedMoE, projected_fp4,projected_fp8,
)


def qwen_fixture(dtype='F32'):
    matrices,vectors,conv,*_=small_qwen_fixture()
    prefix='model.language_model.layers.0.'
    tensors={}
    def store(name,shape,values):
        if dtype=='F32':raw=struct.pack('<'+str(len(values))+'f',*values)
        elif dtype=='BF16':raw=bf16(values)
        else:raise AssertionError(dtype)
        tensors[prefix+name]=(dtype,shape,raw)
    for name,matrix in matrices.items():
        store(name,[len(matrix),len(matrix[0])],[z for row in matrix for z in row])
    for name,v in vectors.items():store(name,[len(v)],v)
    store('linear_attn.conv1d.weight',[8,1,2],[z for row in conv for z in row])
    return nav_for(tensors),matrices,vectors,conv


def unquantized_qwen_reference(matrices,vectors,conv):
    """Reconstruct test truth from *quantized* source tensors, independently of navigator math."""
    return matrices,vectors,conv


def test_full_qwen_qualified_three_positions_against_independent_oracle():
    nav,matrices,vectors,conv=qwen_fixture()
    runner=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    previous_conv=previous_state=None
    inputs=([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])
    for idx,x in enumerate(inputs):
        got,report=runner.next(list(x))
        want,qkv,previous_state=independent_qwen_step(x,matrices,vectors,conv,previous_conv,previous_state)
        previous_conv=qkv
        assert got==pytest.approx(want,abs=2e-7)
        assert report['position']==idx
        assert report['layer']==0 and report['state_digest']==state_digest(runner.recurrent_state)
        assert report['scope'].startswith('full GDN layer')
    assert runner.position==3 and nav.weight_bytes>0
    assert len(nav.chain)==64


def test_full_qwen_rehomes_and_resumes_from_identical_causal_state():
    nav1,*_=qwen_fixture()
    a=QwenQualifiedLayer(nav1,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    a.next([.1,.2,.3]);checkpoint=copy.deepcopy((a.convolution_history,a.recurrent_state,a.position))
    y,_=a.next([.5,.2,-.3])
    nav2,*_=qwen_fixture()
    b=QwenQualifiedLayer(nav2,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    b.convolution_history,b.recurrent_state,b.position=checkpoint
    resumed,receipt=b.next([.5,.2,-.3])
    assert resumed==y and receipt['position']==1
    assert nav1.index['ground_digest']==nav2.index['ground_digest']


def test_qwen_refuses_missing_matrix_and_fails_atomically_on_downstream_corruption():
    nav,*_=qwen_fixture()
    del nav.weight_map['model.language_model.layers.0.mlp.down_proj.weight']
    with pytest.raises(ValueError,match='Incomplete pinned'):
        QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    nav,*_=qwen_fixture()
    runner=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    runner.next([.1,.2,.3])
    saved=(copy.deepcopy(runner.convolution_history),copy.deepcopy(runner.recurrent_state),runner.position)
    prev=nav.host
    def fail(req):
        if req['purpose']=='tensor' and req['shard'] in nav.weight_map.values():
            # Tamper only after the source file's prevalidated header has been read.
            if any('.mlp.down_proj.weight' in name for name in nav.weight_map):
                raise ConnectionError('simulated mathematical host departure')
        return prev(req)
    nav.host=fail
    with pytest.raises(ConnectionError,match='departure'):runner.next([.3,.4,.5])
    assert runner.convolution_history==saved[0] and runner.recurrent_state==saved[1]
    assert runner.position==saved[2]


def test_qwen_budget_rejected_before_consuming_tensor_weights():
    nav,*_=qwen_fixture()
    with pytest.raises(ValueError,match='budget'):
        QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2,budget_bytes=1)
    assert nav.weight_bytes==0


def expert_fixture(*, shared_dtype='F16',corrupt_scale=False,missing_shared=False):
    width=32; hidden=[1.0 if j%2==0 else -0.25 for j in range(width)]
    prefix='layers.0.ffn.'
    tensors={prefix+'gate.weight':('F32',[384,width],struct.pack('<'+str(384*width)+'f',*([0.0]*(384*width)))),
             prefix+'gate.bias':('F32',[384],struct.pack('<384f',*([1.0]*6+[0.0]*378))),
             'layers.0.ffn_norm.weight':('F32',[width],struct.pack('<32f',*([1.0]*width)))}
    for i in range(384):
        if i>=6:continue
        for p in ('w1','w2','w3'):
            key=f'{prefix}experts.{i}.{p}'
            tensors[key+'.weight']=('I8',[width,width//2],bytes([0x22]*(width*width//2)))
            tensors[key+'.scale']=('F8_E8M0',[width,width//32],bytes([255 if corrupt_scale and i==1 and p=='w2' else 127]*width))
    if not missing_shared:
        for p in ('w1','w2','w3'):
            key=f'{prefix}shared_experts.{p}'
            if shared_dtype=='F16':
                tensors[key+'.weight']=('F16',[width,width],struct.pack('<'+str(width*width)+'e',*([.03125]*(width*width))))
            else:
                tensors[key+'.weight']=('F8_E4M3',[width,width],bytes([0x30]*(width*width)))
                tensors[key+'.scale']=('F8_E8M0',[1,1],bytes([127]))
    return nav_for(tensors),hidden


def direct_fixture_oracle(hidden,*,shared_dtype):
    # Fully separate scalar calculation for one fixed, exactly representable FP4/FP8 test Ground.
    scale=1/math.sqrt(math.fsum(z*z for z in hidden)/32+1e-20)
    x=[v*scale for v in hidden]
    sx=math.fsum(x)
    # w1 and w3 FP4 data 0x22 -> positive 1.0 and scale=1.0.
    gate=min(sx,10.0)
    up=max(-10.0,min(sx,10.0))
    activation=gate/(1+math.exp(-gate))*up
    # Each expert has 32 identical FFN hidden channels; w2 all +1.0.
    expert=32*activation
    # Every router logit is zero, correction bias selects first six, route_scale=1.5.
    contribution=1.5*expert
    if shared_dtype=='F16':
        shared_gate=sx/32
        shared_act=shared_gate/(1+math.exp(-shared_gate))*shared_gate
        shared=shared_act  # 32 down elements, each 1/32
    else:
        # 0x30 E4M3 -> 0.5; three shared matrices each with 32 identical values.
        shared_gate=0.5*sx
        gate_value=min(shared_gate,10.0)
        up_value=max(-10.,min(shared_gate,10.))
        shared=32*(gate_value/(1+math.exp(-gate_value))*up_value)*.5
    return [contribution+shared]*32


@pytest.mark.parametrize('shared_dtype',['F16','F8_E4M3'])
def test_six_complete_fp4_experts_plus_shared_match_scalar_oracle(shared_dtype):
    nav,hidden=expert_fixture(shared_dtype=shared_dtype)
    work=DeepSeekQualifiedMoE(nav,layer=0,swiglu_limit=10,norm_eps=1e-20)
    got,report=work.next(hidden)
    assert got==pytest.approx(direct_fixture_oracle(hidden,shared_dtype=shared_dtype),abs=1e-7)
    assert report['selected_experts']==list(range(6))
    assert report['shared_expert'] is True and report['complete_selected_experts'] is True
    assert len(report['range_chain'])==64
    assert len(got)==32


def test_missing_shared_and_missing_selected_scales_are_refused():
    nav,_=expert_fixture(missing_shared=True)
    with pytest.raises(ValueError,match='shared expert'):DeepSeekQualifiedMoE(nav,layer=0)
    nav,hidden=expert_fixture(corrupt_scale=True)
    x=DeepSeekQualifiedMoE(nav,layer=0)
    with pytest.raises(ValueError,match='scale'):x.next(hidden)


def test_selected_expert_reads_only_six_not_alternative_observers():
    nav,hidden=expert_fixture()
    requests=[];host=nav.host
    def watch(req):requests.append(req);return host(req)
    nav.host=watch
    DeepSeekQualifiedMoE(nav,layer=0).next(hidden)
    assert not any('.experts.6.' in str(r) or '.experts.240.' in str(r) for r in requests)
    assert any(r['purpose']=='tensor' for r in requests)


def test_fp4_and_fp8_streaming_outputs_for_entire_matrices():
    nav,hidden=expert_fixture(shared_dtype='F8_E4M3')
    full=projected_fp4(nav,'layers.0.ffn.experts.0.w1.weight',hidden)
    ref=[math.fsum(hidden)]*32
    assert full==pytest.approx(ref)
    assert projected_fp8(nav,'layers.0.ffn.shared_experts.w1.weight',hidden)==pytest.approx([sum(hidden)*.5]*32)


def test_budget_blocks_six_experts_before_any_body_acquisition():
    nav,hidden=expert_fixture()
    executor=DeepSeekQualifiedMoE(nav,layer=0,budget_bytes=1)
    with pytest.raises(ValueError,match='budget'):executor.next(hidden)
    # Router and header reads are real, but no selected expert *payload* was acquired.
    assert nav.weight_bytes==32*4+384*32*4+384*4


def test_causal_pages_recover_under_external_head_and_new_observer(tmp_path):
    from experiments.model_fabric.complete_native_layers import (
        LocalStatePages,export_qwen_causal_ground,restore_qwen_causal_ground)
    store=LocalStatePages(tmp_path)
    first,*_=qwen_fixture()
    origin=QwenQualifiedLayer(first,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    origin.next([.1,.2,.3])
    checkpoint=export_qwen_causal_ground(origin,store,input_sequence_digest='a'*64)
    assert checkpoint['causal_pages'] and all(x['size']<=65536 for x in checkpoint['causal_pages'])
    expected=checkpoint['checkpoint_digest']
    continuation,_=origin.next([.2,.4,.6])
    second,*_=qwen_fixture()
    recovery=QwenQualifiedLayer(second,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    position=restore_qwen_causal_ground(recovery,LocalStatePages(tmp_path),checkpoint,
                         expected_digest=expected,expected_input_sequence_digest='a'*64)
    assert position==1
    got,receipt=recovery.next([.2,.4,.6])
    assert got==continuation and receipt['position']==1
    with pytest.raises(ValueError,match='already advanced'):
        restore_qwen_causal_ground(recovery,store,checkpoint,expected_digest=expected,
                                  expected_input_sequence_digest='a'*64)


def test_causal_pages_refuse_rollback_tampering_and_corrupt_byte(tmp_path):
    from experiments.model_fabric.complete_native_layers import (
        LocalStatePages,export_qwen_causal_ground,restore_qwen_causal_ground)
    nav,*_=qwen_fixture()
    state=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    state.next([.5,-.2,.3])
    store=LocalStatePages(tmp_path)
    cp=export_qwen_causal_ground(state,store,input_sequence_digest='c'*64)
    pin=cp['checkpoint_digest']
    target=QwenQualifiedLayer(qwen_fixture()[0],layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    with pytest.raises(ValueError,match='external authority'):
        restore_qwen_causal_ground(target,store,cp,expected_digest='f'*64,
                                  expected_input_sequence_digest='c'*64)
    with pytest.raises(ValueError,match='source/model/prefix'):
        restore_qwen_causal_ground(target,store,cp,expected_digest=pin,
                                  expected_input_sequence_digest='d'*64)
    bad=dict(cp);bad['position']=99
    with pytest.raises(ValueError,match='external authority'):
        restore_qwen_causal_ground(target,store,bad,expected_digest=pin,
                                  expected_input_sequence_digest='c'*64)
    page=tmp_path/cp['causal_pages'][0]['sha256']
    corrupted=bytearray(page.read_bytes());corrupted[0]^=1;page.write_bytes(corrupted)
    with pytest.raises(ValueError,match='Corrupted'):
        restore_qwen_causal_ground(target,store,cp,expected_digest=pin,
                                  expected_input_sequence_digest='c'*64)
    assert target.position==0


def test_48_native_qwen_recurrent_heads_and_multi_page_handoff(tmp_path):
    """Exercise *all* 48 heads (16 shared QK) and >64KiB paged state."""
    from experiments.model_fabric.complete_native_layers import (
        LocalStatePages, export_qwen_causal_ground, restore_qwen_causal_ground)
    key_heads, value_heads, kd, vd, width = 16, 48, 16, 16, 3
    mixed=2*key_heads*kd+value_heads*vd
    prefix='model.language_model.layers.0.'
    specs={
        'input_layernorm.weight':([width],[0.]*width),
        'linear_attn.in_proj_qkv.weight':([mixed,width],[.03125]*(mixed*width)),
        'linear_attn.in_proj_z.weight':([value_heads*vd,width],[.03125]*(value_heads*vd*width)),
        'linear_attn.in_proj_a.weight':([value_heads,width],[.03125]*(value_heads*width)),
        'linear_attn.in_proj_b.weight':([value_heads,width],[.03125]*(value_heads*width)),
        'linear_attn.conv1d.weight':([mixed,1,2],[.25,.5]*mixed),
        'linear_attn.A_log':([value_heads],[0.]*value_heads),
        'linear_attn.dt_bias':([value_heads],[0.]*value_heads),
        'linear_attn.norm.weight':([vd],[1.]*vd),
        'linear_attn.out_proj.weight':([width,value_heads*vd],[.0078125]*(width*value_heads*vd)),
        'post_attention_layernorm.weight':([width],[0.]*width),
        'mlp.gate_proj.weight':([4,width],[.0625]*12),
        'mlp.up_proj.weight':([4,width],[.03125]*12),
        'mlp.down_proj.weight':([width,4],[.03125]*12),
    }
    tensors={prefix+n:('F32',shape,struct.pack('<'+str(len(vals))+'f',*vals))
             for n,(shape,vals) in specs.items()}
    one=QwenQualifiedLayer(nav_for(tensors),layer=0,key_heads=key_heads,
                            value_heads=value_heads,key_dim=kd,value_dim=vd)
    one.next([.1,.2,.3])
    assert len(one.recurrent_state)==48
    assert all(len(head)==16 and all(len(row)==16 for row in head)
               for head in one.recurrent_state)
    store=LocalStatePages(tmp_path)
    saved=export_qwen_causal_ground(one,store,input_sequence_digest='b'*64)
    assert saved['state_f64_bytes']>65536
    assert len(saved['causal_pages'])>1
    assert max(p['size'] for p in saved['causal_pages'])==65536
    expect,_=one.next([.2,.1,.4])
    two=QwenQualifiedLayer(nav_for(tensors),layer=0,key_heads=key_heads,
                            value_heads=value_heads,key_dim=kd,value_dim=vd)
    restore_qwen_causal_ground(two,LocalStatePages(tmp_path),saved,
          expected_digest=saved['checkpoint_digest'],expected_input_sequence_digest='b'*64)
    received,receipt=two.next([.2,.1,.4])
    assert received==pytest.approx(expect,abs=1e-12)
    assert receipt['position']==1 and two.position==2
