"""General, model-scale-independent *structural* admission and exact weight ranges.

This extension does not turn unfamiliar architectures into Llama computations.
It handles pinned single/multi-shard safetensors locations, exact byte receipts,
operator-family discovery, and a small independently checked RMSNorm probe.
Neither headers nor a model index are tensor values or inference evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter
from typing import Any, Callable

SCHEMA = 'vl64.scale-aware-model-ground.v1'
INDEX_SCHEMA = 'vl64.shard-index.v1'
HEADER_SCHEMA = 'vl64.shard-header.v1'
RANGE_SCHEMA = 'vl64.qualified-shard-range.v1'
MAX_RANGE = 65536
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 32 * 1024 * 1024
DTYPE_BITS = {'BOOL':8,'U8':8,'I8':8,'F4':4,'F6_E2M3':6,
              'F6_E3M2':6,'F8_E4M3':8,'F8_E5M2':8,
              'F8_E8M0':8,'F8_E4M3FNUZ':8,'F8_E5M2FNUZ':8,
              'F16':16,'BF16':16,'I16':16,'U16':16,'F32':32,'I32':32,
              'U32':32,'C64':64,'F64':64,'I64':64,'U64':64}
_REPO = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z')
_REV = re.compile(r'[0-9a-f]{40}\Z')
_SHARD = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def _digest(domain: str, value: Any) -> str:
    return _sha(domain.encode('utf-8')+b'\0'+_canonical(value))


def _pairs(pairs):
    out = {}
    for key,value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key: '+key)
        out[key] = value
    return out


def _json_bytes(raw: bytes, limit: int):
    if not isinstance(raw,bytes) or not 2 <= len(raw) <= limit:
        raise ValueError('Unbounded or missing metadata bytes')
    value=json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,
                     parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value,dict):
        raise ValueError('Metadata must be an object')
    return value


def _identifier(model, revision):
    if not isinstance(model,str) or not _REPO.fullmatch(model) or not isinstance(revision,str) or not _REV.fullmatch(revision):
        raise ValueError('Model must have a pinned immutable revision')


def admit_index(index_raw: bytes, *, model: str, revision: str):
    """Admit exact bytes of a sharded weight_map (not a weight availability proof)."""
    _identifier(model,revision)
    index=_json_bytes(index_raw,MAX_INDEX_BYTES)
    mapping=index.get('weight_map')
    if not isinstance(mapping,dict) or not mapping or len(mapping)>1_000_000:
        raise ValueError('Missing or oversized sharded weight map')
    if any(not isinstance(k,str) or not k or len(k)>512 or
           not isinstance(v,str) or not _SHARD.fullmatch(v)
           for k,v in mapping.items()):
        raise ValueError('Unsafe tensor name or shard path')
    size=index.get('metadata',{}).get('total_size')
    if size is not None and (type(size) not in (int,float) or not math.isfinite(size)
                             or not 0<=size<2**63 or int(size)!=size):
        raise ValueError('Unqualified advertised payload size')
    ground={'schema':INDEX_SCHEMA,'model':model,'revision':revision,
            'index_sha256':_sha(index_raw),'tensor_count':len(mapping),
            'shard_count':len(set(mapping.values())),
            'advertised_payload_bytes':None if size is None else int(size),
            'weight_values_observed':False}
    ground['ground_digest']=_digest(INDEX_SCHEMA,ground)
    return ground,mapping


def admit_shard_header(header_raw: bytes, *, filename: str, expected_names: set[str],
                       file_size: int | None = None):
    """Validate all tensor extents and the index/header bijection for one shard."""
    if not isinstance(filename,str) or not _SHARD.fullmatch(filename):
        raise ValueError('Unsafe shard name')
    if not isinstance(header_raw,bytes) or len(header_raw)<10:
        raise ValueError('Missing safetensors header')
    n=struct.unpack('<Q',header_raw[:8])[0]
    if not 2<=n<=MAX_HEADER_BYTES or len(header_raw)!=n+8:
        raise ValueError('Incomplete or oversized safetensors header')
    fields=_json_bytes(header_raw[8:],MAX_HEADER_BYTES)
    fields.pop('__metadata__',None)
    if set(fields)!=expected_names:
        raise ValueError('Shard header and index tensor inventories differ')
    if len(fields)>1_000_000:
        raise ValueError('Oversized per-shard tensor inventory')
    spans=[]; descriptors={}
    for name,item in fields.items():
        if not isinstance(item,dict) or set(item)!={'dtype','shape','data_offsets'}:
            raise ValueError('Malformed tensor definition')
        dtype=item['dtype']; shape=item['shape']; offsets=item['data_offsets']
        if dtype not in DTYPE_BITS or not isinstance(shape,list) or len(shape)>8 or any(
                type(dim) is not int or dim<0 or dim>2**48 for dim in shape):
            raise ValueError('Unadmitted shape or dtype')
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x) is not int or x<0 for x in offsets):
            raise ValueError('Unadmitted tensor offsets')
        start,end=offsets
        if end<start or end-start!=(math.prod(shape)*DTYPE_BITS[dtype]+7)//8:
            raise ValueError('Declared tensor byte extent disagrees with shape')
        if file_size is not None and (type(file_size) is not int or end+n+8>file_size):
            raise ValueError('Tensor exceeds observed shard size')
        spans.append((start,end))
        descriptors[name]={'dtype':dtype,'shape':shape,'data_offsets':offsets,
                           'file':filename,'header_bytes':len(header_raw)}
    prev=0
    for start,end in sorted(spans):
        if start<prev:
            raise ValueError('Overlapping safetensors tensor offsets')
        prev=end
    ground={'schema':HEADER_SCHEMA,'file':filename,
            'header_sha256':_sha(header_raw),'header_bytes':len(header_raw),
            'tensor_count':len(descriptors),'payload_span_bytes':prev,
            'file_bytes':file_size,'weight_values_observed':False}
    ground['ground_digest']=_digest(HEADER_SCHEMA,ground)
    return ground,descriptors


def _family(config):
    kind=config.get('model_type')
    if kind=='llama':
        tc=config; layer_types=['causal_gqa']*tc.get('num_hidden_layers',0)
        families={'rms_norm','causal_gqa','rotary','swiglu','vocab_projection'}
        status='NUMERIC_ADAPTER_AVAILABLE_FOR_ADMITTED_BF16_F32_WEIGHTS'
    elif kind=='qwen3_5':
        tc=config.get('text_config',{})
        layer_types=tc.get('layer_types',[])
        families={'rms_norm','swiglu','vocab_projection','vision_encoder',
                  'gated_delta_linear_attention','causal_gqa','partial_multimodal_rope',
                  'gated_attention_output','depthwise_conv_state'}
        status='STRUCTURE_ONLY_SPECIALIZED_ATTENTION_AND_VISION_ADAPTERS_REQUIRED'
    elif kind=='deepseek_v41':
        tc=config.get('text_config',{})
        layer_types=['compressed_sparse_attention_moe']*tc.get('num_hidden_layers',0)
        families={'rms_norm','compressed_sparse_attention','mla_projection','yarn_rope',
                  'moe_routing','engram_lookup','mHC','FP8_UE8M0','FP4_expert',
                  'vision_encoder','vocab_projection'}
        status='STRUCTURE_ONLY_MOE_SPARSE_ATTENTION_AND_QUANTIZED_ADAPTERS_REQUIRED'
    else:
        raise ValueError('Unknown model family: refuse inferred execution')
    count=tc.get('num_hidden_layers')
    if type(count) is not int or not 1<=count<=512 or len(layer_types)!=count:
        raise ValueError('Invalid or incomplete layer schedule')
    h=tc.get('hidden_size')
    if type(h) is not int or not 1<=h<=262144:
        raise ValueError('Invalid hidden width')
    return kind,tc,layer_types,families,status


def admit_architecture(config_raw: bytes, *, model: str, revision: str, index_ground: dict):
    _identifier(model,revision)
    config=_json_bytes(config_raw,MAX_INDEX_BYTES)
    if index_ground.get('model')!=model or index_ground.get('revision')!=revision:
        raise ValueError('Model/index revision mismatch')
    if index_ground.get('ground_digest')!=_digest(INDEX_SCHEMA,{k:v for k,v in index_ground.items() if k!='ground_digest'}):
        raise ValueError('Modified index Ground')
    kind,tc,types,ops,status=_family(config)
    stages=[{'layer':i,'family':t,'requires_numeric_adapter':kind!='llama'} for i,t in enumerate(types)]
    ground={'schema':SCHEMA,'model':model,'revision':revision,
            'index_digest':index_ground['ground_digest'], 'config_sha256':_sha(config_raw),
            'model_type':kind,'num_layers':len(stages),'hidden_size':tc['hidden_size'],
            'layer_schedule':stages,'operator_families':sorted(ops),
            'architecture_status':status,'shard_count':index_ground['shard_count'],
            'tensor_count':index_ground['tensor_count'],
            'advertised_payload_bytes':index_ground['advertised_payload_bytes'],
            'weight_payload_bytes_observed':0,'full_model_executed':False,
            'multimodal':kind in ('qwen3_5','deepseek_v41')}
    ground['ground_digest']=_digest(SCHEMA,ground)
    return ground


class QualifiedShardNavigator:
    """One index, independently admitted headers, exact request/response witnesses."""
    def __init__(self, *, model: str, revision: str, index_raw: bytes,
                 range_host: Callable[[dict],dict], approved_hosts: set[str]):
        self.index,self.weight_map=admit_index(index_raw,model=model,revision=revision)
        if not callable(range_host) or not approved_hosts:
            raise ValueError('No qualified range host')
        self.host=range_host
        self.approved=set(approved_hosts)
        self.headers={};self.tensors={}
        self.range_count=0;self.weight_bytes=0
        self.chain='0'*64

    def _read_range(self, shard, begin, end, *, purpose):
        if shard not in set(self.weight_map.values()) or type(begin) is not int or type(end) is not int or not 0<=begin<end or end-begin>MAX_RANGE:
            raise ValueError('Unqualified shard range request')
        req={'model':self.index['model'],'revision':self.index['revision'],
             'index_ground_digest':self.index['ground_digest'],'shard':shard,
             'byte_begin':begin,'byte_end_exclusive':end,'purpose':purpose}
        reply=self.host(dict(req))
        if not isinstance(reply,dict) or any(reply.get(k)!=v for k,v in req.items()) or reply.get('status')!='QUALIFIED_EXACT_SHARD_RANGE' or reply.get('observer_host') not in self.approved:
            raise ValueError('Observer-host receipt not qualified')
        raw=reply.get('payload')
        if not isinstance(raw,bytes) or len(raw)!=end-begin or reply.get('sha256')!=_sha(raw):
            raise ValueError('Incomplete or altered shard range')
        witness={**req,'observer_host':reply['observer_host'],'sha256':reply['sha256']}
        disposition=reply.get('cache_disposition')
        if disposition == 'PINNED_GLANCE_DERIVATION':
            # New qualified *local* observer, NOT a fresh network observation.
            # Its source Ground and the exact original fragments become part
            # of this execution's range evidence chain.
            head=reply.get('glance_ground_head')
            origins=reply.get('original_witnesses')
            if (not isinstance(head,str) or len(head)!=64 or
                    any(z not in '0123456789abcdef' for z in head) or
                    not isinstance(origins,list) or not origins or len(origins)>65536):
                raise ValueError('Unqualified reconstructed observation lineage')
            witness['cache_disposition']=disposition
            witness['glance_ground_head']=head
            witness['historical_sources_digest']=_digest(
                'vl64.glance-derivation-witness.v1',origins)
        elif disposition is not None:
            if disposition not in ('FRESH_HOST_OBSERVATION',
                                   'RETAINED_ORIGINAL_OBSERVATION',
                                   'REQUALIFIED_EQUAL_BYTES'):
                raise ValueError('Unadmitted cache evidence disposition')
            witness['cache_disposition']=disposition
        self.chain=_digest(RANGE_SCHEMA,{'previous':self.chain,'witness':witness})
        self.range_count+=1
        if purpose=='tensor':self.weight_bytes+=len(raw)
        return raw

    def header(self, shard):
        if shard in self.headers: return self.headers[shard]
        if shard not in set(self.weight_map.values()):
            raise ValueError('Shard not present in index')
        head=self._read_range(shard,0,8,purpose='header')
        length=struct.unpack('<Q',head)[0]
        if not 2<=length<=MAX_HEADER_BYTES:
            raise ValueError('Unbounded shard header')
        parts=[head]
        for pos in range(8,8+length,MAX_RANGE):
            parts.append(self._read_range(shard,pos,min(8+length,pos+MAX_RANGE),purpose='header'))
        raw=b''.join(parts)
        names={k for k,v in self.weight_map.items() if v==shard}
        ground,descriptors=admit_shard_header(raw,filename=shard,expected_names=names)
        self.headers[shard]=ground
        self.tensors.update(descriptors)
        return ground

    def locate(self,tensor):
        if tensor not in self.weight_map:
            raise ValueError('Tensor not present in signed/pinned index')
        shard=self.weight_map[tensor]
        self.header(shard)
        return self.tensors[tensor]

    def get_raw(self,tensor,*,flat_start=0,count=None):
        desc=self.locate(tensor)
        total=math.prod(desc['shape']); dtype=desc['dtype']
        if count is None:count=total-flat_start
        if type(flat_start) is not int or type(count) is not int or flat_start<0 or count<=0 or flat_start+count>total:
            raise ValueError('Tensor selection exceeds its typed extent')
        bits=DTYPE_BITS[dtype]
        if flat_start*bits%8 or (count*bits%8 and flat_start+count!=total):
            raise ValueError('Packed dtype range must preserve byte alignment')
        begin=desc['header_bytes']+desc['data_offsets'][0]+flat_start*bits//8
        stop=begin+(count*bits+7)//8
        result=bytearray()
        for p in range(begin,stop,MAX_RANGE):
            result.extend(self._read_range(desc['file'],p,min(p+MAX_RANGE,stop),purpose='tensor'))
        return bytes(result),desc

    def small_numeric_probe(self):
        """Execute an exact real BF16/F32 norm *if* its bytes can be admitted.

        This never infers weights from a config and is not full-model inference.
        """
        names=sorted((k for k in self.weight_map if 'norm' in k.casefold() and
                      k.endswith('.weight')), key=lambda x:(
                          '.layers.0.' not in x, x))
        # A bounded candidate search avoids traversing every MoE shard on a
        # failed probe. Selecting by name never supplies numerical evidence.
        for name in names[:64]:
            desc=self.locate(name)
            if desc['dtype'] not in ('BF16','F32') or len(desc['shape'])!=1 or not 1<=desc['shape'][0]<=32768:
                continue
            raw,_=self.get_raw(name)
            if desc['dtype']=='BF16':
                w=[struct.unpack('<f',struct.pack('<I',b<<16))[0] for (b,) in struct.iter_unpack('<H',raw)]
            else:
                w=list(struct.unpack('<'+'f'*desc['shape'][0],raw))
            if not all(math.isfinite(v) for v in w):
                raise ValueError('Nonfinite observed pretrained weights')
            n=len(w); x=[math.sin((i+1)*0.07)*0.25 for i in range(n)]; eps=1e-6
            # Separate numerical expressions, rather than treating metadata as inference.
            avg=sum(z*z for z in x)/n
            got=[z*y/math.sqrt(avg+eps) for z,y in zip(x,w)]
            scale=1/math.sqrt(math.fsum(z*z for z in x)/n+eps)
            reference=[x[i]*scale*w[i] for i in range(n)]
            error=max(abs(a-b) for a,b in zip(got,reference))
            if error>1e-7:raise ValueError('Independent norm check diverged')
            return {'status':'WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED',
                    'model':self.index['model'],'revision':self.index['revision'],
                    'tensor':name,'dimension':n,'actual_weight_bytes':len(raw),
                    'max_abs_error':round(error,12),
                    'range_witness_chain_digest':self.chain,
                    'scope':'one pretrained normalization tensor, not full-model inference'}
        return {'status':'UNRESOLVED_NO_ADMITTED_BF16_F32_NORMALIZATION_TENSOR',
                'scope':'no claim of weight-backed model execution'}
