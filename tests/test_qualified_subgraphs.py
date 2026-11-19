"""Sharded trained subgraph adapter: exact ranges, real-valued router, refusal."""
import hashlib
import math
import struct
import pytest

from test_architecture_probe import nav_for,bf16
from experiments.model_fabric.qualified_subgraphs import (
    observed_values, observed_rows, probe_deepseek_trained_router,
)


def test_bf16_f16_f32_qualified_matrix_slices_and_true_offsets():
    tensors={
        'x.bf16':('BF16',[3,4],bf16([.5,1.,2.,-1.]*3)),
        'x.f16':('F16',[3,4],struct.pack('<12e',*[.5,-1.,2.,3.]*3)),
        'x.f32':('F32',[3,4],struct.pack('<12f',*[.1,.2,.3,.4]*3)),
    }
    nav=nav_for(tensors)
    x=[1.,2.,3.,4.]
    assert observed_rows(nav,'x.bf16',x,first=1,count=2)==pytest.approx([4.5,4.5])
    assert observed_rows(nav,'x.f16',x,first=1,count=2)==pytest.approx([16.5,16.5])
    assert observed_rows(nav,'x.f32',x,first=1,count=2)==pytest.approx([3.,3.])
    assert observed_values(nav,'x.f16',start=4,count=4)==pytest.approx([.5,-1.,2.,3.])
    assert nav.range_count>4 and nav.weight_bytes>0


def test_real_384_router_fixture_only_selected_six_and_weight_names():
    width=4
    logits=[((i%31)-15)*.001 for i in range(384)]
    # Every row has known first element. Synthetic input is cos(j*.013)*.0625;
    # no assumptions about 'trained' data from the fixture itself.
    matrix=[0.]*(384*width)
    for i in range(384):matrix[i*width]=logits[i]
    selection=[0.]*384
    selection[17]=3.0
    tensors={
      'layers.0.ffn.gate.weight':('F16',[384,width],struct.pack('<'+str(len(matrix))+'e',*matrix)),
      'layers.0.ffn.gate.bias':('F32',[384],struct.pack('<384f',*selection)),
    }
    for i in range(384):
        for part in ('w1','w2','w3'):
            tensors[f'layers.0.ffn.experts.{i}.{part}.weight']=('BF16',[1,1],bf16([.1]))
    nav=nav_for(tensors)
    out=probe_deepseek_trained_router(nav)
    assert out['status']=='REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED'
    assert 17 in out['selected_experts'] and len(out['selected_experts'])==6
    assert len(out['selected_weight_paths'])==6
    assert sum(len(v) for v in out['selected_weight_paths'].values())==18
    assert out['weight_bytes']==(384*width*2+384*4)
    assert len(out['range_chain'])==64
    assert out['max_abs_independent_error']<1e-10


def test_router_refuses_when_bias_or_selected_path_missing():
    w=('F16',[384,2],struct.pack('<768e',*([0.]*768)))
    nav=nav_for({'layers.0.ffn.gate.weight':w})
    with pytest.raises(ValueError,match='bias'):
        probe_deepseek_trained_router(nav)
    nav=nav_for({'layers.0.ffn.gate.weight':w,
                 'layers.0.ffn.gate.bias':('F32',[384],struct.pack('<384f',*([0.]*384)))})
    with pytest.raises(ValueError,match='Selected expert absent'):
        probe_deepseek_trained_router(nav)


def test_unqualified_numeric_dtype_refused_instead_of_fake_scale():
    nav=nav_for({'quant':('F8_E4M3',[2,2],b'\x38'*4)})
    with pytest.raises(ValueError,match='Unsupported'):
        observed_values(nav,'quant')


def test_shape_bound_row_selection_refuses_truncation():
    nav=nav_for({'m':('F32',[2,3],struct.pack('<6f',1,2,3,4,5,6))})
    with pytest.raises(ValueError,match='shape'):observed_rows(nav,'m',[1.,2.])
    with pytest.raises(ValueError,match='Unqualified'):observed_rows(nav,'m',[1.,2.,3.],first=1,count=2)


def test_six_selected_fp4_expert_slices_weight_and_scale_actual_numeric():
    from experiments.model_fabric.qualified_subgraphs import probe_deepseek_selected_fp4_expert_slices
    selected=[9,15,4,20,31,0]
    tensors={}
    for i in selected:
        for path in ('w1','w3','w2'):
            prefix=f'layers.0.ffn.experts.{i}.{path}'
            tensors[prefix+'.weight']=('F4',[32,32],bytes([0x22]*512))
            tensors[prefix+'.scale']=('F8_E8M0',[32,1],bytes([127]*32))
    nav=nav_for(tensors)
    report=probe_deepseek_selected_fp4_expert_slices(nav,{
        'status':'REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED',
        'layer':0,'selected_experts':selected})
    assert report['status']=='REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED'
    assert [v['expert'] for v in report['slices']]==selected
    assert report['max_abs_independent_error']==pytest.approx(0.,abs=1e-12)
    # 6 experts x (2 full input rows + 1 down row) with exact E8M0 scales.
    assert nav.weight_bytes==6*3*(16+1)


def test_selected_fp4_missing_scale_and_nan_scale_fail_closed():
    from experiments.model_fabric.qualified_subgraphs import probe_deepseek_selected_fp4_expert_slices
    selected=list(range(6)); tensors={}
    for i in selected:
        for part in ('w1','w3','w2'):
            prefix=f'layers.0.ffn.experts.{i}.{part}'
            tensors[prefix+'.weight']=('F4',[32,32],bytes([0x22]*512))
            tensors[prefix+'.scale']=('F8_E8M0',[32,1],bytes([127]*32))
    tensor='layers.0.ffn.experts.2.w2.scale'
    corrupted=dict(tensors);corrupted.pop(tensor)
    route={'status':'REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED','layer':0,'selected_experts':selected}
    with pytest.raises(ValueError,match='scale'):
        probe_deepseek_selected_fp4_expert_slices(nav_for(corrupted),route)
    corrupted=dict(tensors);corrupted[tensor]=('F8_E8M0',[32,1],bytes([255]*32))
    with pytest.raises(ValueError,match='scale'):
        probe_deepseek_selected_fp4_expert_slices(nav_for(corrupted),route)
