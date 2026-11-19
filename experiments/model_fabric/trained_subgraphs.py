"""Typed mathematical *subgraphs* for Qwen Gated DeltaNet and DeepSeek routed FFN.

Data/transport interfaces are supplied by the caller.  No manifest, architecture
name, or checkpoint reference is evidence that the required numerical weights
were retrieved.  Modules live outside the 64-KiB VL64 core.

Qwen3.5/3.8 have two distinct RMSNorm contracts: standard backbone RMSNorm
uses (1 + weight) while GatedDeltaNet RMSNormGated uses weight directly.
"""
from __future__ import annotations

import math

from .architecture_ops import (
    _finite, _sigmoid, qwen_gates, depthwise_causal_conv_step,
    qwen_gated_delta_step, qwen_gated_rms_norm, deepseek_sqrtsoftplus_route,
    routed_expert_mix, state_digest,
)


def centered_rms_norm(x, weight, *, eps=1e-6):
    """Qwen3.5 zero-centered RMSNorm: norm(x) * (1 + weight)."""
    if not x or len(x) != len(weight) or eps <= 0:
        raise ValueError('Invalid centered RMS normalization dimension')
    _finite(x, 'centered norm input'); _finite(weight, 'centered norm weights')
    factor=1.0/math.sqrt(math.fsum(v*v for v in x)/len(x)+eps)
    return [z*factor*(1.0+w) for z,w in zip(x,weight)]


def standard_rms_norm(x, weight, *, eps=1e-6):
    """Standard RMSNorm used in the DeepSeek FFN path."""
    if not x or len(x) != len(weight) or eps <= 0:
        raise ValueError('Invalid RMS normalization dimension')
    _finite(x,'norm input');_finite(weight,'norm weight')
    factor=1/math.sqrt(math.fsum(z*z for z in x)/len(x)+eps)
    return [z*factor*w for z,w in zip(x,weight)]


def dense_matvec(matrix, x):
    """Pure scalar reference-oriented linear projection, no implicit bias."""
    if not x or not matrix or any(len(row)!=len(x) for row in matrix):
        raise ValueError('Unqualified linear projection dimension')
    _finite(x,'linear input')
    result=[]
    for row in matrix:
        _finite(row,'linear weights')
        result.append(math.fsum(a*b for a,b in zip(row,x)))
    _finite(result,'linear output')
    return result


def swiglu_expert(hidden,gate_weight,up_weight,down_weight):
    """The exact dense expert MLP: down(silu(gate(x)) * up(x))."""
    g=dense_matvec(gate_weight,hidden)
    u=dense_matvec(up_weight,hidden)
    if len(g)!=len(u):
        raise ValueError('Expert gate/up widths differ')
    intermediate=[z*_sigmoid(z)*v for z,v in zip(g,u)]
    return dense_matvec(down_weight,intermediate)


def qwen_gated_delta_layer(hidden, projection, vectors, convolution,
                           *, key_heads, value_heads, key_dim, value_dim,
                           conv_history=None, recurrent_state=None, eps=1e-6):
    """Execute one complete Qwen3.5-family GDN+SwiGLU transformer layer.

    ``projection(name, x)`` must return the COMPLETE value of named matrix @ x,
    and ``vectors(name)`` returns admitted 1-D tensors. ``convolution`` has
    [2*nk*dk+nv*dv, kernel] channel-major coefficients. They must all come
    from a qualified model Ground in weight-backed use. Outputs are immutable:
    previous history and recurrent state are not mutated if later work fails.
    This method handles a *linear-attention layer*, not the alternating full-
    attention layers, vision path, or 64-layer Qwen model.
    """
    if (not hidden or any(type(v) is not int or v<=0 for v in
            (key_heads,value_heads,key_dim,value_dim)) or value_heads%key_heads):
        raise ValueError('Unadmitted Qwen linear-attention architecture')
    _finite(hidden,'Qwen input hidden')
    width=len(hidden); k_total=key_heads*key_dim; v_total=value_heads*value_dim
    mixed_width=2*k_total+v_total
    if len(convolution)!=mixed_width or any(not isinstance(row,(list,tuple)) for row in convolution):
        raise ValueError('Incomplete GatedDelta convolution')
    normed=centered_rms_norm(hidden,vectors('input_layernorm.weight'),eps=eps)
    raw_qkv=projection('linear_attn.in_proj_qkv.weight',normed)
    z=projection('linear_attn.in_proj_z.weight',normed)
    a=projection('linear_attn.in_proj_a.weight',normed)
    b=projection('linear_attn.in_proj_b.weight',normed)
    if (len(raw_qkv)!=mixed_width or len(z)!=v_total or
            len(a)!=value_heads or len(b)!=value_heads):
        raise ValueError('Qwen trained projections have incompatible extents')
    convolved,next_conv=depthwise_causal_conv_step(
        raw_qkv,convolution,conv_history,activation='silu')
    q=convolved[:k_total];k=convolved[k_total:2*k_total];v=convolved[2*k_total:]
    beta,decay=qwen_gates(a,b,vectors('linear_attn.A_log'),vectors('linear_attn.dt_bias'))
    qh=[q[i*key_dim:(i+1)*key_dim] for i in range(key_heads)]
    kh=[k[i*key_dim:(i+1)*key_dim] for i in range(key_heads)]
    vh=[v[i*value_dim:(i+1)*value_dim] for i in range(value_heads)]
    recurrent,next_state=qwen_gated_delta_step(qh,kh,vh,decay,beta,recurrent_state)
    norm_weight=vectors('linear_attn.norm.weight')
    if len(norm_weight)!=value_dim:
        raise ValueError('Internal GDN norm must be per value head')
    heads=[qwen_gated_rms_norm(row,z[i*value_dim:(i+1)*value_dim],
                               norm_weight,eps=eps) for i,row in enumerate(recurrent)]
    attention=projection('linear_attn.out_proj.weight',[z for head in heads for z in head])
    if len(attention)!=width:raise ValueError('Qwen GDN output width differs')
    residual=[x+y for x,y in zip(hidden,attention)]
    post=centered_rms_norm(residual,vectors('post_attention_layernorm.weight'),eps=eps)
    gate=projection('mlp.gate_proj.weight',post)
    up=projection('mlp.up_proj.weight',post)
    if not gate or len(gate)!=len(up):raise ValueError('Qwen SwiGLU width differs')
    nonlinear=[g*_sigmoid(g)*u for g,u in zip(gate,up)]
    down=projection('mlp.down_proj.weight',nonlinear)
    if len(down)!=width:raise ValueError('Qwen FFN output width differs')
    result=[r+d for r,d in zip(residual,down)]
    _finite(result,'Qwen complete layer output')
    state={'convolution':next_conv,'delta':next_state}
    return result,state,{'state_digest':state_digest(next_state),
                         'conv_digest':state_digest(next_conv),
                         'layer_type':'qwen3_5.gated_delta_swiglu',
                         'input_width':width,'output_width':len(result)}


def deepseek_moe_layer(hidden, norm_weight, router_matrix, selection_bias,
                       experts, shared, *, topk=6, route_scale=1.5, eps=1e-6):
    """Complete routed FFN numerical subgraph with selected-only expert access.

    Each ``experts(i, normalized_hidden)`` returns exactly one expert's complete
    numerical SwiGLU output; ``shared(normalized_hidden)`` returns the shared
    expert output. No unselected expert is called. This does not imply that
    DeepSeek's mHC, Engram or compressed sparse attention has run.
    """
    n=standard_rms_norm(hidden,norm_weight,eps=eps)
    logits=dense_matvec(router_matrix,n)
    route=deepseek_sqrtsoftplus_route(logits,selection_bias,
                                      topk=topk,route_scale=route_scale)
    seen=[]
    def selected(i):
        seen.append(i)
        return experts(i,n)
    mixed=routed_expert_mix(route,selected,shared(n))
    if len(mixed)!=len(hidden):raise ValueError('MoE output width differs')
    out=[x+y for x,y in zip(hidden,mixed)]
    _finite(out,'DeepSeek FFN result')
    return out,{'selected_experts':route['selected_experts'],
                'routing_weights':route['routing_weights'],
                'consulted_experts':seen,
                'scope':'one routed FFN subgraph; no mHC or full decoder'}


def required_deepseek_expert_tensors(index_names, layer, expert_ids):
    """Plan only selected weight AND scale tensors; refuse missing companions.

    Accept only exact identifier paths present in the observed index. The
    returned plan does not fetch any weights, and it cannot substitute absent
    scale tensors. Alias-free paths keep independent observer identity simple.
    """
    if (type(layer) is not int or layer<0 or
            not expert_ids or len(set(expert_ids))!=len(expert_ids) or
            any(type(i) is not int or not 0<=i<384 for i in expert_ids)):
        raise ValueError('Unqualified DeepSeek expert selection')
    names=set(index_names);out=[]
    for i in expert_ids:
        stage=[]
        for part in ('w1','w3','w2'):
            prefix=f'layers.{layer}.ffn.experts.{i}.{part}'
            name=prefix+'.weight';scale=prefix+'.scale'
            if name not in names or scale not in names:
                raise ValueError('Incomplete expert weight/scale identity '+prefix)
            stage.append((name,scale))
        out.append({'expert':i,'pairs':stage})
    return out
