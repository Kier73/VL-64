"""Complete typed, *qualified* native layer/FFN execution at arbitrary scales.

The caller supplies the observed sharded index and qualified range host. A
manifest alone is never execution. Numeric execution occurs only after every
required output coordinate has been computed with its exact source weights.
Neither an architecture label nor a successful toy fixture confers a real
pretrained inference claim. All code lives outside the VL-64 64 KiB core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture_ops import (_sigmoid, deepseek_sqrtsoftplus_route,
                               mx_block_matvec, state_digest)
from .qualified_subgraphs import observed_values, observed_rows
from .trained_subgraphs import (centered_rms_norm, standard_rms_norm,
                                qwen_gated_delta_layer)
from .scaled_model_fabric import QualifiedShardNavigator

DENSE_DTYPES = {'BF16', 'F16', 'F32'}
QUANT_DTYPES = {'F4', 'I8'}
BITS = {'BF16': 16, 'F16': 16, 'F32': 32, 'F4': 4, 'I8': 8, 'F8_E4M3': 8, 'F8_E8M0':8}
QWEN_PARTS = ('input_layernorm.weight', 'linear_attn.in_proj_qkv.weight',
              'linear_attn.in_proj_z.weight', 'linear_attn.in_proj_a.weight',
              'linear_attn.in_proj_b.weight', 'linear_attn.conv1d.weight',
              'linear_attn.A_log', 'linear_attn.dt_bias',
              'linear_attn.norm.weight', 'linear_attn.out_proj.weight',
              'post_attention_layernorm.weight', 'mlp.gate_proj.weight',
              'mlp.up_proj.weight', 'mlp.down_proj.weight')


def _extent(nav, name):
    d = nav.locate(name)
    if d['dtype'] not in BITS:
        raise ValueError('Unimplemented qualified tensor dtype: ' + name)
    return d, math.prod(d['shape']) * BITS[d['dtype']] // 8


def _positive_budget(value):
    if type(value) is not int or value < 1:
        raise ValueError('Positive finite acquisition byte budget required')


def _f32_state_digest(vector):
    if not vector or not all(type(x) in (int, float) and math.isfinite(x) for x in vector):
        raise ValueError('Unadmitted numeric hidden vector')
    return state_digest(vector)


class QwenQualifiedLayer:
    """Run a complete trained GDN+SwiGLU layer, with atomic causal state.

    Uses the existing *full* mathematical Qwen layer implementation and the
    existing exact-range navigator, without separately materializing 27B model
    matrices. Each projector streams rows and returns every output element.
    Only GDN layers are admitted; full-attention/vision requires other adapters.
    """

    def __init__(self, nav, *, layer, key_heads, value_heads, key_dim, value_dim,
                 budget_bytes=1_500_000_000, eps=1e-6):
        if not isinstance(nav, QualifiedShardNavigator):
            raise ValueError('QualifiedShardNavigator required')
        if type(layer) is not int or layer < 0 or any(type(z) is not int or z < 1
                for z in (key_heads, value_heads, key_dim, value_dim)):
            raise ValueError('Invalid native GDN layer coordinate')
        if value_heads % key_heads:
            raise ValueError('Invalid GDN grouped head topology')
        _positive_budget(budget_bytes)
        self.nav, self.layer = nav, layer
        self.kh, self.vh, self.kd, self.vd = key_heads, value_heads, key_dim, value_dim
        self.eps = eps
        self.prefix = f'model.language_model.layers.{layer}.'
        self.names = {part: self.prefix + part for part in QWEN_PARTS}
        if any(name not in nav.weight_map for name in self.names.values()):
            raise ValueError('Incomplete pinned Qwen layer weight inventory')
        self.shapes = {}
        extent = 0
        for part,name in self.names.items():
            desc,nbytes = _extent(nav,name)
            if desc['dtype'] not in DENSE_DTYPES:
                raise ValueError('Qwen dense projection cannot decode this dtype')
            self.shapes[part] = desc['shape']
            extent += nbytes
        hidden = self.shapes['input_layernorm.weight'][0]
        k_total = key_heads * key_dim
        v_total = value_heads * value_dim
        mixed = 2*k_total + v_total
        intermediate = self.shapes['mlp.gate_proj.weight'][0]
        expected = {
          'input_layernorm.weight':[hidden], 'linear_attn.in_proj_qkv.weight':[mixed,hidden],
          'linear_attn.in_proj_z.weight':[v_total,hidden],
          'linear_attn.in_proj_a.weight':[value_heads,hidden],
          'linear_attn.in_proj_b.weight':[value_heads,hidden],
          'linear_attn.conv1d.weight':[mixed,1,self.shapes['linear_attn.conv1d.weight'][-1]],
          'linear_attn.A_log':[value_heads], 'linear_attn.dt_bias':[value_heads],
          'linear_attn.norm.weight':[value_dim],
          'linear_attn.out_proj.weight':[hidden,v_total],
          'post_attention_layernorm.weight':[hidden],
          'mlp.gate_proj.weight':[intermediate,hidden],
          'mlp.up_proj.weight':[intermediate,hidden],
          'mlp.down_proj.weight':[hidden,intermediate],
        }
        if any(self.shapes[p] != shape for p,shape in expected.items()):
            raise ValueError('Qwen layer weight shapes are not composable')
        if not 1 <= self.shapes['linear_attn.conv1d.weight'][-1] <= 64:
            raise ValueError('Unqualified GDN convolution kernel length')
        if extent > budget_bytes:
            raise ValueError('Qwen complete-layer acquisition exceeds explicit byte budget')
        self.hidden, self.estimated_weight_bytes = hidden, extent
        self.convolution_history = None
        self.recurrent_state = None
        self.position = 0
        self._vectors = {p: observed_values(nav,self.names[p]) for p in
                         ('input_layernorm.weight','linear_attn.A_log','linear_attn.dt_bias',
                          'linear_attn.norm.weight','post_attention_layernorm.weight')}
        kernel=self.shapes['linear_attn.conv1d.weight'][-1]
        conv=observed_values(nav,self.names['linear_attn.conv1d.weight'])
        self._convolution=[conv[i:i+kernel] for i in range(0,len(conv),kernel)]

    def next(self, hidden):
        if len(hidden) != self.hidden:
            raise ValueError('Hidden vector does not match pinned layer width')
        def project(part, vector):
            if part not in self.names or part not in self.shapes or len(self.shapes[part])!=2:
                raise ValueError('Unadmitted trained projection')
            return observed_rows(self.nav,self.names[part],vector)
        def vector(part):
            if part not in self._vectors:
                raise ValueError('Unadmitted native vector tensor')
            return self._vectors[part]
        output, next_state, evidence = qwen_gated_delta_layer(
            hidden, project, vector, self._convolution,
            key_heads=self.kh, value_heads=self.vh, key_dim=self.kd,
            value_dim=self.vd, conv_history=self.convolution_history,
            recurrent_state=self.recurrent_state, eps=self.eps)
        # Never advance causal coordinates or publish a partial state when
        # range acquisition, math, or downstream projection fails.
        self.convolution_history = next_state['convolution']
        self.recurrent_state = next_state['delta']
        self.position += 1
        return output, {'schema':'vl64.qualified-qwen-layer.v1',
                        'scope':'full GDN layer only, no full-attention/vision path',
                        'model':self.nav.index['model'],'revision':self.nav.index['revision'],
                        'layer':self.layer, 'position':self.position-1,
                        'hidden_digest':_f32_state_digest(output),
                        'state_digest':evidence['state_digest'],
                        'convolution_digest':evidence['conv_digest'],
                        'range_chain':self.nav.chain,
                        'weight_bytes_read':self.nav.weight_bytes}


def _fp4_shapes(nav, name):
    if name not in nav.weight_map:
        raise ValueError('Selected or shared expert weight missing: '+name)
    desc,_ = _extent(nav,name)
    if len(desc['shape'])!=2 or desc['dtype'] not in QUANT_DTYPES:
        raise ValueError('Selected expert requires packed E2M1 matrix')
    if desc['dtype']=='I8' and '.ffn.experts.' not in name:
        raise ValueError('I8 packed interpretation only admitted for routed expert bodies')
    rows, phys = desc['shape']
    cols = phys * (2 if desc['dtype']=='I8' else 1)
    if cols%32 or cols < 32 or rows < 1:
        raise ValueError('Packed FP4 matrix is not K32 aligned')
    scale = name.removesuffix('.weight') + '.scale'
    if scale not in nav.weight_map:
        raise ValueError('Missing matching E8M0 scale: '+name)
    sc = nav.locate(scale)
    if sc['dtype'] != 'F8_E8M0' or sc['shape'] != [rows,cols//32]:
        raise ValueError('FP4 scale grid mismatches weight coordinates')
    return rows, cols, phys, scale


def projected_fp4(nav,name,x,*,chunk_bytes=32768):
    """Stream complete E2M1 matrix-vector through the qualified reader.

    Exact tensor bytes and their E8M0 row scales are acquired independently.
    Neither payload nor output matrix size has a model-specific ceiling.
    """
    if not isinstance(nav, QualifiedShardNavigator):
        raise ValueError('Qualified shard observer required')
    rows,cols,physical,scale_name = _fp4_shapes(nav,name)
    if len(x)!=cols or any(not math.isfinite(t) for t in x):
        raise ValueError('Selected expert input width invalid')
    if type(chunk_bytes) is not int or not 1<=chunk_bytes<=65536:
        raise ValueError('Invalid bounded chunk budget')
    row_bytes=physical+cols//32
    take=max(1,chunk_bytes//row_bytes)
    result=[]
    for first in range(0,rows,take):
        n=min(take,rows-first)
        raw,_=nav.get_raw(name,flat_start=first*physical,count=n*physical)
        scales,_=nav.get_raw(scale_name,flat_start=first*(cols//32),count=n*(cols//32))
        result.extend(mx_block_matvec(raw,scales,x,rows=n,cols=cols,
                            format='mxfp4_e2m1',scale_block=(1,32)))
    if len(result)!=rows or not all(math.isfinite(z) for z in result):
        raise ValueError('Incomplete FP4 expert projection')
    return result


def _expert_names(prefix):
    return {part:prefix+part+'.weight' for part in ('w1','w3','w2')}


def _fp8_shapes(nav, name):
    d,_=_extent(nav,name)
    if len(d['shape']) != 2 or d['dtype']!='F8_E4M3':
        raise ValueError('Expected native FP8 E4M3 expert matrix')
    rows,cols=d['shape']
    scale=name.removesuffix('.weight')+'.scale'
    sc=nav.locate(scale)
    if sc['dtype'] != 'F8_E8M0' or sc['shape'] != [(rows+31)//32,(cols+31)//32]:
        raise ValueError('FP8 expert 32x32 scale grid differs')
    return rows,cols,scale


def projected_fp8(nav,name,x,*,batch_rows=32):
    """E4M3 row windows with separately admitted 32x32 exponent scales."""
    rows,cols,scale_name=_fp8_shapes(nav,name)
    if len(x)!=cols or batch_rows!=32:
        raise ValueError('Unadmitted FP8 expert dimensions or alignment')
    ncol=(cols+31)//32
    result=[]
    for first in range(0,rows,32):
        n=min(32,rows-first)
        raw,_=nav.get_raw(name,flat_start=first*cols,count=n*cols)
        scales,_=nav.get_raw(scale_name,flat_start=(first//32)*ncol,count=ncol)
        result.extend(mx_block_matvec(raw,scales,x,rows=n,cols=cols,
                          format='mxfp8_e4m3',scale_block=(32,32)))
    if len(result)!=rows or not all(math.isfinite(z) for z in result):
        raise ValueError('Incomplete pretrained FP8 shared projection')
    return result


def _expert_dimensions(nav, names, *, packed):
    if packed:
        shapes={part:_fp4_shapes(nav,name)[:2] for part,name in names.items()}
    else:
        shapes={}
        for part,name in names.items():
            if name not in nav.weight_map:
                raise ValueError('Missing shared expert tensor: '+name)
            d,_=_extent(nav,name)
            if d['dtype'] == 'F8_E4M3':
                shapes[part]=_fp8_shapes(nav,name)[:2]
            elif d['dtype'] in DENSE_DTYPES and len(d['shape'])==2:
                shapes[part]=tuple(d['shape'])
            else:
                raise ValueError('Unsupported native shared expert dtype/shape')
    a,b,c=shapes['w1'],shapes['w3'],shapes['w2']
    if a!=b or c!=(a[1],a[0]):
        raise ValueError('Uncomposable expert gate/up/down dimensions')
    return a[1],a[0]


def _swiglu(gate,up,limit):
    if not 0 <= limit <= 1000:
        raise ValueError('Invalid native DeepSeek SwiGLU limit')
    if len(gate)!=len(up) or not gate:
        raise ValueError('Incomplete expert gate/up projection')
    if limit>0:
        gate=[min(z,limit) for z in gate]
        up=[max(-limit,min(limit,z)) for z in up]
    return [(z*_sigmoid(z))*u for z,u in zip(gate,up)]


class DeepSeekQualifiedMoE:
    """Complete routed MoE FFN with *only* six selected FP4 experts + shared.

    Architecture-specific mHC/Engram/CSA remain separate. Input is the layer's
    post-attention residual; norm applies once. Router bias chooses expert IDs
    but never alters the six normalized contributions.
    """
    def __init__(self, nav, *, layer, budget_bytes=800_000_000,
                 swiglu_limit=10.0, norm_eps=1e-20, topk=6, route_scale=1.5):
        if not isinstance(nav, QualifiedShardNavigator) or type(layer) is not int or layer<0:
            raise ValueError('Qualified DeepSeek layer/observer required')
        _positive_budget(budget_bytes)
        self.nav,self.layer,self.budget=nav,layer,budget_bytes
        self.limit,self.eps,self.topk,self.route_scale=swiglu_limit,norm_eps,topk,route_scale
        self.prefix=f'layers.{layer}.ffn.'
        self.norm_name=f'layers.{layer}.ffn_norm.weight'
        self.router_name=self.prefix+'gate.weight'
        self.bias_name=self.prefix+'gate.bias'
        self.shared_names=_expert_names(self.prefix+'shared_experts.')
        for name in (self.norm_name,self.router_name,self.bias_name):
            if name not in nav.weight_map:raise ValueError('Missing native DeepSeek router/norm '+name)
        self.width=nav.locate(self.norm_name)['shape'][0]
        r=nav.locate(self.router_name)
        if r['shape']!=[384,self.width] or r['dtype'] not in DENSE_DTYPES:
            raise ValueError('Unadmitted 384-way DeepSeek router')
        if nav.locate(self.bias_name)['shape']!=[384]:
            raise ValueError('Unadmitted DeepSeek routing bias')
        if any(n not in nav.weight_map for n in self.shared_names.values()):
            raise ValueError('DeepSeek shared expert required')
        shared_input,_=_expert_dimensions(nav,self.shared_names,packed=False)
        if shared_input!=self.width:
            raise ValueError('Shared expert input width mismatch')

    def _project_expert(self, names, x, *, packed):
        proj=(lambda name,v:projected_fp4(self.nav,name,v)) if packed else (
            lambda name,v:(projected_fp8(self.nav,name,v) if self.nav.locate(name)['dtype']=='F8_E4M3' else observed_rows(self.nav,name,v)))
        gate=proj(names['w1'],x);up=proj(names['w3'],x)
        return proj(names['w2'],_swiglu(gate,up,self.limit))

    def next(self, hidden):
        if len(hidden)!=self.width or any(not math.isfinite(z) for z in hidden):
            raise ValueError('DeepSeek MoE requires full admitted residual vector')
        norm=observed_values(self.nav,self.norm_name)
        x=standard_rms_norm(hidden,norm,eps=self.eps)
        router_logits=observed_rows(self.nav,self.router_name,x)
        bias=observed_values(self.nav,self.bias_name)
        route=deepseek_sqrtsoftplus_route(router_logits,bias,topk=self.topk,
                                         route_scale=self.route_scale)
        names=[];weight_budget=0
        for expert in route['selected_experts']:
            tensors=_expert_names(self.prefix+f'experts.{expert}.')
            inp,_=_expert_dimensions(self.nav,tensors,packed=True)
            if inp!=self.width:raise ValueError('Selected expert hidden width mismatch')
            names.append((expert,tensors))
        # Before fetching even one selected expert body, check all six selected
        # and shared matrix extents against the externally approved byte budget.
        for _,tensors in names:
            for name in tensors.values():
                d,extent=_extent(self.nav,name)
                weight_budget+=extent
                weight_budget+=math.prod(self.nav.locate(name.removesuffix('.weight')+'.scale')['shape'])
        for name in self.shared_names.values():
            d,extent=_extent(self.nav,name);weight_budget+=extent
            if d['dtype']=='F8_E4M3':
                _,scale_extent=_extent(self.nav,name.removesuffix('.weight')+'.scale')
                weight_budget+=scale_extent
        if weight_budget>self.budget:
            raise ValueError('Complete selected MoE exceeds explicit byte budget')
        shared=self._project_expert(self.shared_names,x,packed=False)
        output=list(shared)
        for (expert,tensors),factor in zip(names,route['routing_weights']):
            values=self._project_expert(tensors,x,packed=True)
            if len(values)!=self.width:raise ValueError('Incomplete selected expert result')
            output=[s+factor*v for s,v in zip(output,values)]
        if len(output)!=self.width or not all(math.isfinite(v) for v in output):
            raise ValueError('Unqualified complete mixture output')
        return output, {'schema':'vl64.qualified-deepseek-moe.v1',
                        'scope':'trained post-norm FFN only; no CSA/Engram/mHC',
                        'model':self.nav.index['model'],'revision':self.nav.index['revision'],
                        'layer':self.layer,'selected_experts':route['selected_experts'],
                        'routing_weights':route['routing_weights'],
                        'output_digest':_f32_state_digest(output),
                        'weight_bytes_read':self.nav.weight_bytes,
                        'range_chain':self.nav.chain,
                        'shared_expert':True,'complete_selected_experts':True}

# Content-addressed causal state is separate from both weight observation and
# the 64-KiB executable core. An externally retained head digest is required
# for rollback/equivocation refusal: self-hashing an untrusted checkpoint is
# not authentication.
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path


class LocalStatePages:
    """Content-addressed 64 KiB pages persisted atomically in a chosen directory.

    One trusted writer, not a distributed consensus service. Replicate the
    pages separately and keep the expected head with an authenticated caller.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data):
        if not isinstance(data,bytes) or not 1<=len(data)<=65536:
            raise ValueError('Causal page outside 64 KiB range')
        sha=hashlib.sha256(data).hexdigest()
        target=self.root / sha
        if target.is_file():
            if target.read_bytes()!=data:raise ValueError('Existing causal page is corrupt')
        else:
            fd,name=tempfile.mkstemp(dir=self.root,prefix='.pending-')
            try:
                with os.fdopen(fd,'wb') as stream:
                    stream.write(data);stream.flush();os.fsync(stream.fileno())
                os.replace(name,target)
            finally:
                if os.path.exists(name):os.unlink(name)
        return {'sha256':sha,'size':len(data)}

    def get(self, descriptor):
        if (not isinstance(descriptor,dict) or set(descriptor)!={'sha256','size'} or
                not isinstance(descriptor['sha256'],str) or
                len(descriptor['sha256'])!=64 or
                any(ch not in '0123456789abcdef' for ch in descriptor['sha256']) or
                type(descriptor['size']) is not int or not 1<=descriptor['size']<=65536):
            raise ValueError('Invalid causal page descriptor')
        raw=(self.root/descriptor['sha256']).read_bytes()
        if len(raw)!=descriptor['size'] or hashlib.sha256(raw).hexdigest()!=descriptor['sha256']:
            raise ValueError('Corrupted causal state page')
        return raw


def _record_digest(body):
    wire=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(b'vl64.qualified-native-orbit.v1\0'+wire).hexdigest()


def _admitted_digest(v):
    return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


def export_qwen_causal_ground(runner, store, *, input_sequence_digest):
    """Publish exact F64 causal state; caller must separately retain head pin."""
    if not isinstance(runner,QwenQualifiedLayer) or not isinstance(store,LocalStatePages):
        raise ValueError('Qualified Qwen layer and page store required')
    if runner.position==0 or not _admitted_digest(input_sequence_digest):
        raise ValueError('No completed input prefix/externally named lineage')
    conv=runner.convolution_history
    delta=runner.recurrent_state
    mixed=2*runner.kh*runner.kd+runner.vh*runner.vd
    if (not isinstance(conv,list) or not isinstance(delta,list) or
            len(delta)!=runner.vh or any(len(mat)!=runner.kd for mat in delta) or
            any(len(row)!=runner.vd for mat in delta for row in mat) or
            any(len(row)!=mixed for row in conv)):
        raise ValueError('Qwen causal state shape changed or incomplete')
    values=[float(z) for row in conv for z in row]
    values.extend(float(z) for mat in delta for row in mat for z in row)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite causal state')
    raw=struct.pack('<'+'d'*len(values),*values)
    pages=[store.put(raw[i:i+65536]) for i in range(0,len(raw),65536)]
    body={'schema':'vl64.qualified-native-orbit.v1',
          'model':runner.nav.index['model'],'revision':runner.nav.index['revision'],
          'index_ground_digest':runner.nav.index['ground_digest'],
          'layer':runner.layer,'position':runner.position,
          'input_sequence_digest':input_sequence_digest,
          'key_heads':runner.kh,'value_heads':runner.vh,'key_dim':runner.kd,'value_dim':runner.vd,
          'convolution_history_rows':len(conv),'causal_pages':pages,
          'state_f64_bytes':len(raw), 'causal_state_digest':state_digest([conv,delta])}
    return {**body,'checkpoint_digest':_record_digest(body)}


def restore_qwen_causal_ground(runner,store,checkpoint,*,expected_digest,
                               expected_input_sequence_digest):
    """Resume on a new observer only with external model, lineage & head pins."""
    if not isinstance(runner,QwenQualifiedLayer) or not isinstance(store,LocalStatePages):
        raise ValueError('Qualified receiver and page store required')
    if runner.position!=0:
        raise ValueError('Cannot overwrite an already advanced observer orbit')
    if not isinstance(checkpoint,dict) or not _admitted_digest(expected_digest) or not _admitted_digest(expected_input_sequence_digest):
        raise ValueError('Unpinned Qwen checkpoint')
    body={k:v for k,v in checkpoint.items() if k!='checkpoint_digest'}
    if (checkpoint.get('checkpoint_digest')!=expected_digest or
            _record_digest(body)!=expected_digest):
        raise ValueError('Checkpoint head differs from external authority pin')
    required={'schema','model','revision','index_ground_digest','layer','position',
              'input_sequence_digest','key_heads','value_heads','key_dim','value_dim',
              'convolution_history_rows','causal_pages','state_f64_bytes','causal_state_digest'}
    if set(body)!=required or body['schema']!='vl64.qualified-native-orbit.v1':
        raise ValueError('Unsupported Qwen orbit checkpoint fields')
    if (body['model']!=runner.nav.index['model'] or body['revision']!=runner.nav.index['revision'] or
            body['index_ground_digest']!=runner.nav.index['ground_digest'] or
            body['layer']!=runner.layer or body['input_sequence_digest']!=expected_input_sequence_digest or
            body['key_heads']!=runner.kh or body['value_heads']!=runner.vh or
            body['key_dim']!=runner.kd or body['value_dim']!=runner.vd):
        raise ValueError('Incompatible source/model/prefix or layer receiver')
    count=body['convolution_history_rows'];mixed=2*runner.kh*runner.kd+runner.vh*runner.vd
    max_history=runner.shapes['linear_attn.conv1d.weight'][-1]-1
    if (type(count) is not int or not 0<=count<=max_history or
            type(body['position']) is not int or body['position']<1 or
            not isinstance(body['causal_pages'],list) or not body['causal_pages']):
        raise ValueError('Invalid causal coordinate/extent')
    total=count*mixed+runner.vh*runner.kd*runner.vd
    if body['state_f64_bytes']!=8*total or total>2**27:
        raise ValueError('Declared causal state extent incorrect')
    raw=b''.join(store.get(page) for page in body['causal_pages'])
    if len(raw)!=8*total:
        raise ValueError('Incomplete causal page chain')
    vals=struct.unpack('<'+'d'*total,raw)
    if not all(math.isfinite(z) for z in vals):
        raise ValueError('Nonfinite recovered causal state')
    conv=[list(vals[i*mixed:(i+1)*mixed]) for i in range(count)]
    cursor=count*mixed;delta=[]
    for head in range(runner.vh):
        mat=[]
        for row in range(runner.kd):
            mat.append(list(vals[cursor:cursor+runner.vd]));cursor+=runner.vd
        delta.append(mat)
    if body['causal_state_digest'] != state_digest([conv,delta]):
        raise ValueError('Recovered state differs from published causal digest')
    runner.convolution_history=conv
    runner.recurrent_state=delta
    runner.position=body['position']
    return runner.position
