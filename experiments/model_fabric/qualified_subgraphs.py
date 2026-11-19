"""Actual sharded pretrained numeric subgraphs through qualified observer ranges.

A model index grants coordinates, not execution. Every required tensor range is
requested through ``QualifiedShardNavigator`` and checked against the observed
shard header. Probes consume real trained weights with explicitly synthetic
hidden inputs: *not* full pretrained inference or complete multimodal layers.
"""
from __future__ import annotations

import math
import struct

from .architecture_ops import (
    _sigmoid, qwen_gates, depthwise_causal_conv_step,
    qwen_gated_delta_step, qwen_gated_rms_norm,
    deepseek_sqrtsoftplus_route,
)
from .trained_subgraphs import centered_rms_norm
from .scaled_model_fabric import QualifiedShardNavigator


def _decode_float(raw, dtype):
    if dtype=='BF16':
        if len(raw)%2:raise ValueError('Misaligned BF16')
        out=[struct.unpack('<f',struct.pack('<I',b<<16))[0] for (b,) in struct.iter_unpack('<H',raw)]
    elif dtype=='F16':
        if len(raw)%2:raise ValueError('Misaligned F16')
        out=[v[0] for v in struct.iter_unpack('<e',raw)]
    elif dtype=='F32':
        if len(raw)%4:raise ValueError('Misaligned F32')
        out=[v[0] for v in struct.iter_unpack('<f',raw)]
    else:raise ValueError('Tensor dtype requires a dedicated quantized decoder: '+dtype)
    if not all(math.isfinite(z) for z in out):
        raise ValueError('Nonfinite pretrained weight')
    return out


def observed_values(nav,name,*,start=0,count=None):
    if not isinstance(nav,QualifiedShardNavigator):
        raise ValueError('Qualified sharded observer required')
    desc=nav.locate(name)
    if desc['dtype'] not in ('BF16','F16','F32'):
        raise ValueError('Unsupported dense numeric dtype')
    raw,_=nav.get_raw(name,flat_start=start,count=count)
    return _decode_float(raw,desc['dtype'])


def observed_rows(nav,name,x,*,first=0,count=None):
    """Acquire only indicated, exact matrix rows in bounded native ranges."""
    desc=nav.locate(name)
    shape=desc['shape']
    if len(shape)!=2 or not x or shape[1]!=len(x):
        raise ValueError('Matrix row projection is not shape-admitted')
    if count is None:count=shape[0]-first
    if (type(first) is not int or type(count) is not int or
            first<0 or count<=0 or first+count>shape[0]):
        raise ValueError('Unqualified projected row partition')
    cols=len(x);result=[]
    # One-row-at-a-time memory bound, but allow coalescing adjacent rows to
    # amortize range-provider calls without changing the 64KiB message cap.
    chunk=max(1,min(64,65536//(cols*2)))
    for low in range(first,first+count,chunk):
        take=min(chunk,first+count-low)
        flat=observed_values(nav,name,start=low*cols,count=take*cols)
        for row in range(take):
            weights=flat[row*cols:(row+1)*cols]
            result.append(math.fsum(w*z for w,z in zip(weights,x)))
    return result


def probe_qwen_trained_delta_head(nav, *, layer=0, head=0):
    """Real Qwen layer: norm -> trained Q/K/V -> conv -> gated delta -> gate norm.

    A single V head is computed, plus its corresponding Q/K head and one
    admitted output-projection row contribution. Other V heads and downstream
    dense FFN are NOT executed: the receipt is a trained *subgraph*.
    """
    prefix=f'model.language_model.layers.{layer}.'
    names=[prefix+s for s in ('input_layernorm.weight','linear_attn.in_proj_qkv.weight',
           'linear_attn.in_proj_z.weight','linear_attn.in_proj_a.weight',
           'linear_attn.in_proj_b.weight','linear_attn.conv1d.weight',
           'linear_attn.A_log','linear_attn.dt_bias','linear_attn.norm.weight',
           'linear_attn.out_proj.weight')]
    missing=[n for n in names if n not in nav.weight_map]
    if missing:raise ValueError('Unadmitted Qwen trained head tensors: '+','.join(missing[:2]))
    norm_name,qkv_name,z_name,a_name,b_name,conv_name,alog_name,dt_name,gamma_name,out_name=names
    qdesc=nav.locate(qkv_name)
    if len(qdesc['shape'])!=2 or qdesc['shape'][0]!=10240 or qdesc['shape'][1]!=5120 or not 0<=head<48:
        raise ValueError('Qwen 27B GatedDelta shape differs from pinned architecture')
    norm=observed_values(nav,norm_name)
    if len(norm)!=5120:raise ValueError('Qwen input norm width')
    hidden=[math.sin((j+1)*.017)*.125 for j in range(5120)]
    normed=centered_rms_norm(hidden,norm)
    qk_head=head//3
    locations=[qk_head*128,2048+qk_head*128,4096+head*128]
    samples=[observed_rows(nav,qkv_name,normed,first=offset,count=128)
             for offset in locations]
    convdesc=nav.locate(conv_name)
    if convdesc['shape']!=[10240,1,4]:
        raise ValueError('Incorrect native Qwen depthwise conv shape')
    convweights=[observed_values(nav,conv_name,start=offset*4,count=128*4)
                 for offset in locations]
    convrows=[[raw[(c*4)+t] for t in range(4)] for raw in convweights for c in range(128)]
    convolved,_=depthwise_causal_conv_step([v for group in samples for v in group],convrows,activation='silu')
    a=observed_rows(nav,a_name,normed,first=head,count=1)
    b=observed_rows(nav,b_name,normed,first=head,count=1)
    alog=observed_values(nav,alog_name,start=head,count=1)
    dt=observed_values(nav,dt_name,start=head,count=1)
    beta,decay=qwen_gates(a,b,alog,dt)
    res,delta=qwen_gated_delta_step([convolved[:128]],[convolved[128:256]],
                                     [convolved[256:]],decay,beta)
    gamma=observed_values(nav,gamma_name)
    if len(gamma)!=128:raise ValueError('Qwen internal gated norm width')
    gate=observed_rows(nav,z_name,normed,first=head*128,count=128)
    head_out=qwen_gated_rms_norm(res[0],gate,gamma)
    # A true partial output contribution, not the complete 5120-dim vector.
    # out_proj has shape [5120,6144]; request only its first row and the 128
    # columns corresponding to this V head; no hidden contributions fabricated.
    outdesc=nav.locate(out_name)
    if outdesc['shape']!=[5120,6144]:raise ValueError('Qwen output projection shape')
    out_frag=observed_values(nav,out_name,start=head*128,count=128)
    partial=math.fsum(x*w for x,w in zip(head_out,out_frag))
    # Different scalar realization from the exact observed fragments, without
    # calling the composed Qwen mathematical operators.
    q_raw,k_raw,v_raw=samples
    cos_conv=[]
    for block in range(3):
        vals=convweights[block]
        cos_conv.extend([(vals[c*4+3]*samples[block][c])*_sigmoid(vals[c*4+3]*samples[block][c]) for c in range(128)])
    def n_l2(x):return [z/math.sqrt(math.fsum(t*t for t in x)+1e-6) for z in x]
    qn=n_l2(cos_conv[:128]);kn=n_l2(cos_conv[128:256]);vv=cos_conv[256:]
    zz=a[0]+dt[0]
    softplus=zz+math.log1p(math.exp(-zz)) if zz>=0 else math.log1p(math.exp(zz))
    decay_ref=-math.exp(alog[0])*softplus
    beta_ref=_sigmoid(b[0])
    val_ref=[beta_ref*math.fsum(qn[j]*kn[j]*vv[c] for j in range(128))/math.sqrt(128) for c in range(128)]
    rms=1/math.sqrt(math.fsum(t*t for t in val_ref)/128+1e-6)
    out_ref=[t*rms*gamma[c]*gate[c]*_sigmoid(gate[c]) for c,t in enumerate(val_ref)]
    partial_ref=math.fsum(x*w for x,w in zip(out_ref,out_frag))
    error=max(abs(partial-partial_ref),abs(beta[0]-beta_ref),abs(decay[0]-decay_ref))
    if not math.isfinite(error) or error>2e-5:
        raise ValueError('Genuine Qwen trained head disagrees with independent scalar reference')
    return {'schema':'vl64.trained-architecture-subgraph.v1',
            'status':'REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED',
            'model':nav.index['model'],'revision':nav.index['revision'],
            'layer':layer,'value_head':head,'input_scope':'synthetic_hidden',
            'trained_operations':['zero_centered_rmsnorm','QKV_rows','depthwise_conv',
                 'delta_gate','gated_recurrent_head','gated_rmsnorm','partial_out_projection'],
            'output_row0_head_contribution':round(partial,12),
            'max_abs_independent_error':round(error,12),
            'weight_bytes':nav.weight_bytes,'range_count':nav.range_count,
            'range_chain':nav.chain,
            'scope':'one trained recurrent head, not a full Qwen layer or next-token inference'}


def probe_deepseek_trained_router(nav, *, layer=0, topk=6):
    """Real trained 384-way DeepSeek router; plan exactly selected expert paths.

    This verifies the trained router and its selected tensor identities; no
    expert matmul is claimed until all selected weight/scale bytes are loaded.
    """
    prefix=f'layers.{layer}.ffn.gate.'
    weight_name=prefix+'weight';bias_name=prefix+'bias'
    if weight_name not in nav.weight_map or bias_name not in nav.weight_map:
        raise ValueError('Pinned DeepSeek FFN router weights or selection bias missing')
    desc=nav.locate(weight_name)
    if len(desc['shape'])!=2 or desc['shape'][0]!=384 or not 1<=desc['shape'][1]<=32768:
        raise ValueError('DeepSeek router is not 384-way qualified tensor')
    width=desc['shape'][1]
    hidden=[math.cos((j+1)*.013)*.0625 for j in range(width)]
    bias=observed_values(nav,bias_name)
    if len(bias)!=384:raise ValueError('DeepSeek router selection bias shape')
    logits=observed_rows(nav,weight_name,hidden,count=384)
    route=deepseek_sqrtsoftplus_route(logits,bias,topk=topk)
    # Entirely separate reference using a log1p-based stable softplus branch.
    scores=[math.sqrt(math.log1p(math.exp(x)) if x<0 else x+math.log1p(math.exp(-x))) for x in logits]
    selected=sorted(range(384),key=lambda i:(-(scores[i]+bias[i]),i))[:topk]
    denom=math.fsum(scores[i] for i in selected)
    weights=[1.5*scores[i]/denom for i in selected]
    error=max(abs(x-y) for x,y in zip(weights,route['routing_weights']))
    if selected!=route['selected_experts'] or error>2e-8:
        raise ValueError('Real trained routing differs from independent selector')
    # Only selected experts' names enter the plan; exact per-tensor header
    # admission happens before numerical dispatch, not by assuming names.
    selected_paths={str(i):[f'layers.{layer}.ffn.experts.{i}.{p}.weight'
             for p in ('w1','w3','w2')] for i in selected}
    missing=[p for paths in selected_paths.values() for p in paths if p not in nav.weight_map]
    if missing:raise ValueError('Selected expert absent from observed index: '+missing[0])
    return {'schema':'vl64.trained-architecture-subgraph.v1',
            'status':'REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED',
            'model':nav.index['model'],'revision':nav.index['revision'],
            'layer':layer,'input_scope':'synthetic_hidden',
            'selected_experts':selected,'routing_weights':[round(x,12) for x in weights],
            'selected_weight_paths':selected_paths,
            'max_abs_independent_error':round(error,12),
            'weight_bytes':nav.weight_bytes,'range_count':nav.range_count,
            'range_chain':nav.chain,
            'scope':'real 384-way trained router; selected experts not numerically executed'}
