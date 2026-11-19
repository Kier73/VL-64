"""Mocked HTTP-206 tests for pinned large-model observation (no network)."""
import io
import json
import struct
from urllib import request
from unittest.mock import Mock

import pytest
from experiments.model_fabric import scaled_model_lab as lab
from test_scaled_model_fabric import (bf16, json_bytes, make_shard, qwen, deepseek)

class Response:
    def __init__(self,data,url,*,status=200,headers=None):
        self.data=io.BytesIO(data);self.url=url;self.status=status;self.headers=headers or {}
    def __enter__(self):return self
    def __exit__(self,*args):self.data.close()
    def read(self,n):return self.data.read(n)
    def geturl(self):return self.url

@pytest.mark.parametrize('selection,model_config', [('qwen',qwen),('deepseek',deepseek)])
def test_metadata_and_small_numeric_probe_from_qualified_https_ranges(monkeypatch,selection,model_config):
    name='model.layers.0.input_layernorm.weight'
    weights=bf16([1.0+(i%5)/64 for i in range(32)])
    shard='model-00001-of-00048.safetensors'
    file=make_shard({name:([32],'BF16',weights)})
    index=json_bytes({'metadata':{'total_size':len(weights)},'weight_map':{name:shard}})
    config=json_bytes(model_config())
    seen=[]
    def urlopen(req,timeout):
        url=req.full_url;seen.append((url,req.headers.get('Range')))
        if url.endswith('config.json'):
            return Response(config,url)
        if url.endswith('model.safetensors.index.json'):
            return Response(index,url)
        header=req.headers.get('Range')
        assert header and header.startswith('bytes=')
        a,b=map(int,header[6:].split('-'))
        return Response(file[a:b+1],url,status=206,headers={'Content-Range':f'bytes {a}-{b}/{len(file)}'})
    monkeypatch.setattr(lab.urllib.request,'urlopen',urlopen)
    report=lab.observe_large_model(selection=selection,numeric=True)
    assert report['layers']==(64 if selection=='qwen' else 40)
    assert report['status']=='STRUCTURAL_MODEL_GROUND_ADMITTED'
    assert report['pretrained_weight_bytes_observed']==len(weights)
    assert report['numeric_probe']['status']=='WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED'
    assert report['numeric_probe']['max_abs_error']<=1e-7
    assert report['exact_http_ranges']==3 # 8 header, JSON header, weight range
    assert report['full_model_executed'] is False
    assert all(url.startswith('https://huggingface.co/') for url,_ in seen)


def test_remote_range_requires_http_206_and_valid_content_range(monkeypatch):
    model,revision=lab.PINNED['qwen']
    fetch=lab.pinned_hf_range_host(model,revision)
    req={'shard':'model-00001.safetensors','byte_begin':100,'byte_end_exclusive':104}
    url=lab._url(model,revision,req['shard'])
    monkeypatch.setattr(lab.urllib.request,'urlopen',lambda *a,**kw:Response(b'1234',url,status=200))
    with pytest.raises(ValueError,match='exact'):
        fetch(req)
    monkeypatch.setattr(lab.urllib.request,'urlopen',lambda *a,**kw:Response(b'1234',url,status=206,
                      headers={'Content-Range':'bytes 0-3/400'}))
    with pytest.raises(ValueError,match='different'):
        fetch(req)


def test_untrusted_redirect_and_unsupported_paths_refused():
    assert not lab._host_allowed('http://huggingface.co/file')
    assert not lab._host_allowed('https://huggingface.co.attacker.com/file')
    assert not lab._host_allowed('https://cas-bridge.xethub.attacker.com/file')
    assert lab._host_allowed('https://cas-bridge.xethub.hf.co/file')
    with pytest.raises(ValueError,match='Unapproved'):
        lab._url(*lab.PINNED['qwen'],'../model.safetensors')
