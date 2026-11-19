"""No-network tests for sharded, arbitrarily sized and non-Llama model structures."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct

import pytest

from experiments.model_fabric.scaled_model_fabric import (
    admit_architecture,admit_index,admit_shard_header,QualifiedShardNavigator,
    MAX_RANGE,_digest,INDEX_SCHEMA,
)

REV='1234567890abcdef1234567890abcdef12345678'
MODEL='example/large-model'


def json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def bf16(values):
    return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',float(v)))[0]>>16) for v in values)


def make_shard(tensors):
    data=b'';header={}
    for name,(shape,dtype,raw) in sorted(tensors.items()):
        start=len(data);data+=raw
        header[name]={'shape':shape,'dtype':dtype,'data_offsets':[start,len(data)]}
    blob=json_bytes(header)
    return struct.pack('<Q',len(blob))+blob+data


def setup(tensors_by_shard):
    shards={f:make_shard(t) for f,t in tensors_by_shard.items()}
    weight_map={k:shard for shard,v in tensors_by_shard.items() for k in v}
    index=json_bytes({'metadata':{'total_size':sum(len(v) for v in shards.values())},'weight_map':weight_map})
    requests=[]
    def host(req):
        requests.append(req)
        raw=shards[req['shard']][req['byte_begin']:req['byte_end_exclusive']]
        return {**req,'status':'QUALIFIED_EXACT_SHARD_RANGE','observer_host':'qualified-local',
                'payload':raw,'sha256':hashlib.sha256(raw).hexdigest()}
    nav=QualifiedShardNavigator(model=MODEL,revision=REV,index_raw=index,
                                range_host=host,approved_hosts={'qualified-local'})
    return nav,requests,shards,index


def qwen():
    return {'model_type':'qwen3_5','text_config':{'model_type':'qwen3_5_text',
        'hidden_size':5120,'num_hidden_layers':64,
        'layer_types':['linear_attention' if i%4!=3 else 'full_attention' for i in range(64)],
        'num_attention_heads':24,'num_key_value_heads':4},
        'vision_config':{'num_hidden_layers':27}}


def deepseek():
    return {'model_type':'deepseek_v41', 'text_config':{'hidden_size':5120,
        'num_hidden_layers':40,'n_routed_experts':384,'num_experts_per_tok':6,
        'rope_scaling':{'rope_type':'yarn'},'engram_layer_ids':[1,14]},
        'quantization_config':{'quant_method':'fp8','expert_dtype':'fp4'}}


def test_both_large_families_admitted_without_claim_of_numeric_execution():
    index=json_bytes({'weight_map':{'model.layers.0.input_layernorm.weight':'model-00001-of-00048.safetensors'}})
    g,_=admit_index(index,model=MODEL,revision=REV)
    a=admit_architecture(json_bytes(qwen()),model=MODEL,revision=REV,index_ground=g)
    assert a['num_layers']==64 and len(a['layer_schedule'])==64
    assert a['layer_schedule'][3]['family']=='full_attention'
    assert 'depthwise_conv_state' in a['operator_families']
    assert a['full_model_executed'] is False and a['weight_payload_bytes_observed']==0
    b=admit_architecture(json_bytes(deepseek()),model=MODEL,revision=REV,index_ground=g)
    assert b['num_layers']==40 and 'engram_lookup' in b['operator_families']
    assert 'FP4_expert' in b['operator_families'] and b['full_model_executed'] is False
    assert a['ground_digest']!=b['ground_digest']


def test_index_rejects_unpinned_unsafe_and_duplicate_mapping():
    with pytest.raises(ValueError,match='pinned'):
        admit_index(json_bytes({'weight_map':{'a':'model-00001.safetensors'}}),model=MODEL,revision='main')
    with pytest.raises(ValueError,match='Unsafe'):
        admit_index(json_bytes({'weight_map':{'a':'../../secret.safetensors'}}),model=MODEL,revision=REV)
    with pytest.raises(ValueError,match='Duplicate'):
        admit_index(b'{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}',model=MODEL,revision=REV)


def test_headers_bind_exact_tensors_shards_ranges_and_hashes():
    norm=bf16([1+i/32 for i in range(16)])
    gamma=struct.pack('<ffff',1.,2.,3.,4.)
    nav,requests,shards,_=setup({'a.safetensors':{'model.layers.0.input_layernorm.weight':([16],'BF16',norm)},
                                 'b.safetensors':{'other.weight':([4],'F32',gamma)}})
    desc=nav.locate('model.layers.0.input_layernorm.weight')
    assert desc['shape']==[16]
    raw,_=nav.get_raw('model.layers.0.input_layernorm.weight')
    assert raw==norm and nav.weight_bytes==len(norm)
    assert nav.get_raw('other.weight',flat_start=2,count=1)[0]==struct.pack('<f',3.)
    assert nav.index['shard_count']==2 and len(nav.headers)==2
    assert all(r['byte_end_exclusive']-r['byte_begin']<=MAX_RANGE for r in requests)
    assert len(nav.chain)==64
    assert nav.small_numeric_probe()['status']=='WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED'


def test_header_must_match_index_and_exact_payload_extents():
    header={'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}
    raw=json_bytes(header);payload=struct.pack('<Q',len(raw))+raw
    with pytest.raises(ValueError,match='inventories'):
        admit_shard_header(payload,filename='a.safetensors',expected_names={'a','b'})
    bad=copy.deepcopy(header)
    bad['a']['data_offsets']=[0,3]
    raw=json_bytes(bad)
    with pytest.raises(ValueError,match='extent'):
        admit_shard_header(struct.pack('<Q',len(raw))+raw,filename='a.safetensors',expected_names={'a'})


def test_rejects_corrupt_range_receipt_and_unknown_host():
    nav,_,_,index=setup({'a.safetensors':{'model.layers.0.input_layernorm.weight':([4],'BF16',bf16([1]*4))}})
    good=nav.host
    def corrupt(req):
        r=good(req);r['sha256']='0'*64;return r
    nav.host=corrupt
    with pytest.raises(ValueError,match='altered'):
        nav.locate('model.layers.0.input_layernorm.weight')
    nav2=QualifiedShardNavigator(model=MODEL,revision=REV,index_raw=index,
                                 range_host=good,approved_hosts={'another-host'})
    with pytest.raises(ValueError,match='qualified'):
        nav2.locate('model.layers.0.input_layernorm.weight')


def test_large_tensor_range_without_materializing_full_tensor():
    huge=2**30
    entry={'model.large.weight':{'dtype':'F32','shape':[huge],
                                  'data_offsets':[0,huge*4]}}
    h=json_bytes(entry);header=struct.pack('<Q',len(h))+h
    index=json_bytes({'weight_map':{'model.large.weight':'a.safetensors'}})
    got=[]
    def host(req):
        begin,end=req['byte_begin'],req['byte_end_exclusive']; got.append(end-begin)
        if begin<len(header): raw=header[begin:end]
        else: raw=struct.pack('<f',1.25)*((end-begin)//4)
        return {**req,'observer_host':'local','status':'QUALIFIED_EXACT_SHARD_RANGE',
                'payload':raw,'sha256':hashlib.sha256(raw).hexdigest()}
    nav=QualifiedShardNavigator(model=MODEL,revision=REV,index_raw=index,
                                range_host=host,approved_hosts={'local'})
    raw,desc=nav.get_raw('model.large.weight',flat_start=huge-1,count=1)
    assert raw==struct.pack('<f',1.25) and desc['shape']==[huge]
    assert max(got)<=MAX_RANGE and nav.weight_bytes==4


def test_chunked_weight_ranges_64k_budget():
    raw=bf16([1]*50000)
    nav,reqs,_,_=setup({'x.safetensors':{'many.weight':([50000],'BF16',raw)}})
    fetched,_=nav.get_raw('many.weight')
    assert fetched==raw and nav.weight_bytes==100000
    data_requests=[r for r in reqs if r['purpose']=='tensor']
    assert len(data_requests)==2 and max(r['byte_end_exclusive']-r['byte_begin'] for r in data_requests)==65536


def test_unsupported_families_cannot_masquerade_as_verified_llama():
    ix,_=admit_index(json_bytes({'weight_map':{'w':'s.safetensors'}}),model=MODEL,revision=REV)
    for model_type in ('qwen3_5_modified','unknown','deepseek_v42'):
        with pytest.raises(ValueError,match='Unknown model'):
            admit_architecture(json_bytes({'model_type':model_type}),model=MODEL,revision=REV,index_ground=ix)
    corrupted=copy.deepcopy(ix);corrupted['tensor_count']=999
    with pytest.raises(ValueError,match='Modified'):
        admit_architecture(json_bytes(qwen()),model=MODEL,revision=REV,index_ground=corrupted)


def test_probe_accepts_non_llama_normalization_name():
    name='model.layers.0.attn_norm.weight'
    raw=bf16([0.5,1.,1.5,2.])
    nav,requests,_,_=setup({'a.safetensors':{name:([4],'BF16',raw)}})
    result=nav.small_numeric_probe()
    assert result['status']=='WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED'
    assert result['tensor']==name and result['actual_weight_bytes']==len(raw)
