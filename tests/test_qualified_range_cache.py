"""Qualified range retention across positions, model families and process restarts."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture, expert_fixture
from test_trained_subgraphs import independent_qwen_step
from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, DeepSeekQualifiedMoE, LocalStatePages,
    export_qwen_causal_ground, restore_qwen_causal_ground,
)
from experiments.model_fabric.qualified_range_cache import QualifiedRangeCache
from experiments.model_fabric.qualified_subgraphs import observed_values


BUDGET = 65_536 * 4


def cached(nav, path, *, host=None, allowed=None, head=None, budget=BUDGET):
    wrapper = QualifiedRangeCache(path, nav.host if host is None else host,
                                  approved_hosts=allowed or {'fixture-host'},
                                  expected_head=head, budget_bytes=budget)
    nav.host = wrapper
    return wrapper


def test_three_position_qwen_reuses_all_admitted_weight_ranges(tmp_path):
    nav, matrices, vectors, conv = qwen_fixture()
    cache = cached(nav, tmp_path)
    runner = QwenQualifiedLayer(nav, layer=0, key_heads=1, value_heads=2,
                                key_dim=2, value_dim=2)
    past_conv = past_state = None
    xs = ([.5, -.2, .3], [-.1, .2, .8], [.7, .6, -.3])
    network_counts = []
    for x in xs:
        got, receipt = runner.next(list(x))
        want, qkv, past_state = independent_qwen_step(x, matrices, vectors, conv,
                                                      past_conv, past_state)
        past_conv = qkv
        assert got == pytest.approx(want, abs=2e-7)
        network_counts.append(cache.stats()['qualified_remote_ranges'])
        assert len(receipt['hidden_digest']) == 64
    assert network_counts[0] > 0
    assert network_counts == [network_counts[0]] * 3
    assert cache.hits >= cache.remote_ranges
    assert cache.stored_bytes <= BUDGET
    assert nav.range_count > cache.remote_ranges  # admitted logical reads vs physical


def test_deepseek_selected_six_and_shared_second_position_uses_zero_new_ranges(tmp_path):
    nav, hidden = expert_fixture(shared_dtype='F8_E4M3')
    cache = cached(nav, tmp_path)
    model = DeepSeekQualifiedMoE(nav, layer=0)
    output, receipt = model.next(hidden)
    first = cache.stats()['qualified_remote_ranges']
    output2, receipt2 = model.next(hidden)
    assert first > 0
    assert cache.remote_ranges == first
    assert output2 == output
    assert receipt2['selected_experts'] == receipt['selected_experts'] == list(range(6))
    assert cache.hits > 0


def test_externally_pinned_catalogue_recovers_to_fresh_navigator(tmp_path):
    tensors = {'x.weight': ('F32', [4], __import__('struct').pack('<4f',1.,2.,3.,4.))}
    nav = nav_for(tensors)
    cache = cached(nav,tmp_path)
    assert observed_values(nav,'x.weight') == [1.,2.,3.,4.]
    head = cache.snapshot()
    assert len(head) == 64
    new_nav = nav_for(tensors)
    def offline(_req):raise AssertionError('Reused cache may not contact remote host')
    retained = cached(new_nav,tmp_path,host=offline,head=head)
    assert observed_values(new_nav,'x.weight') == [1.,2.,3.,4.]
    assert retained.remote_ranges == 0
    assert retained.hits >= 2  # restored header + tensor
    assert retained.stats()['qualified_remote_bytes'] == 0


def test_restart_without_external_head_or_with_wrong_head_is_refused(tmp_path):
    nav=nav_for({'x.weight':('I8',[8],b'abcdefgh')})
    cache=cached(nav,tmp_path)
    nav.get_raw('x.weight')
    head=cache.snapshot()
    with pytest.raises(ValueError,match='externally pinned'):
        cached(nav_for({'x.weight':('I8',[8],b'abcdefgh')}),tmp_path)
    with pytest.raises(ValueError,match='diverges'):
        cached(nav_for({'x.weight':('I8',[8],b'abcdefgh')}),tmp_path,head='0'*64)
    assert head!= '0'*64


def test_corrupted_retained_page_is_refused_even_if_network_is_available(tmp_path):
    nav=nav_for({'w':('I8',[4],b'GOOD')})
    c=cached(nav,tmp_path)
    nav.get_raw('w')
    page=next((tmp_path/'pages').glob(hashlib.sha256(b'GOOD').hexdigest()))
    page.write_bytes(b'EVIL')
    with pytest.raises(ValueError,match='corrupt'):
        nav.get_raw('w')


def test_malformed_host_payload_not_admitted_or_retained(tmp_path):
    nav=nav_for({'w':('I8',[4],b'GOOD')})
    base=nav.host
    def corrupt(req):
        resp=base(req)
        if req['purpose']=='tensor':resp['sha256']='f'*64
        return resp
    c=cached(nav,tmp_path,host=corrupt)
    with pytest.raises(ValueError,match='corrupted'):
        nav.get_raw('w')
    assert c.stored_bytes > 0 # header alone admitted
    assert c.remote_ranges == 2
    assert not (tmp_path/'pages'/hashlib.sha256(b'GOOD').hexdigest()).exists()


def test_rehomed_host_requalified_equal_bytes_and_rejects_divergent(tmp_path):
    tensors={'w':('I8',[4],b'GOOD')}
    nav=nav_for(tensors)
    cache=cached(nav,tmp_path)
    assert nav.get_raw('w')[0]==b'GOOD'
    pin=cache.snapshot()
    nav2=nav_for(tensors)
    base=nav2.host
    def new_host(req):
        reply=base(req)
        reply['observer_host']='new-host'
        return reply
    second=cached(nav2,tmp_path,head=pin,allowed={'new-host'},host=new_host)
    nav2.approved = {'new-host'}
    assert nav2.get_raw('w')[0]==b'GOOD'
    assert second.requalifications >= 2
    assert second.hits == 0
    assert second.remote_ranges >= 2
    # With a changed upstream byte the old externally pinned head must not
    # be republished as if the new observer had verified the same contents.
    nav3=nav_for(tensors)
    previous=nav3.host
    def malicious(req):
        response=previous(req)
        response['observer_host']='new-host'
        if req['purpose']=='tensor':
            response['payload']=b'EVIL'
            response['sha256']=hashlib.sha256(b'EVIL').hexdigest()
        return response
    third=cached(nav3,tmp_path,head=pin,allowed={'new-host'},host=malicious)
    nav3.approved = {'new-host'}
    with pytest.raises(ValueError,match='conflicts'):
        nav3.get_raw('w')


def test_budgeted_cache_eviction_for_nonreuse_does_not_fabricate_hits(tmp_path):
    raw=b'A'*40_000+b'B'*40_000
    nav=nav_for({'w':('I8',[80_000],raw)})
    cache=cached(nav,tmp_path,budget=65_536)
    nav.get_raw('w',count=40_000)
    requests=cache.remote_ranges
    nav.get_raw('w',flat_start=40_000,count=40_000)
    assert cache.remote_ranges > requests
    assert cache.evictions >= 1
    assert cache.stored_bytes<=65_536


def test_exact_request_purpose_and_model_identity_cannot_alias_cached_page(tmp_path):
    nav=nav_for({'w':('I8',[4],b'DATA')})
    c=cached(nav,tmp_path)
    assert nav.get_raw('w')[0]==b'DATA'
    first=c.remote_ranges
    other = nav_for({'w':('I8',[4],b'DATA')})
    other.index['model']='other/fixture-model'
    other.host = c
    # The existing *host* must see the bound request. Even a cache hit cannot
    # be found using a different model identity or purpose.
    with pytest.raises(ValueError,match='[Mm]odel|response'):
        other.get_raw('w')
    assert c.remote_ranges==first
    with pytest.raises(ValueError,match='request fields'):
        c({'shard':'weights.safetensors'})


def test_qwen_causal_checkpoint_and_weight_cache_survive_new_process(tmp_path):
    nav,*_=qwen_fixture()
    cache=cached(nav,tmp_path/'weights')
    a=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    a.next([.1,.2,.3]);saved=export_qwen_causal_ground(
        a,LocalStatePages(tmp_path/'state'),input_sequence_digest='d'*64)
    head=cache.snapshot()
    next_expected,_=a.next([.7,-.2,.9])
    src=Path(__file__).resolve().parents[1]
    script='''
import json,sys
sys.path.insert(0, sys.argv[1]+'/tests')
from test_complete_native_layers import qwen_fixture
from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer,LocalStatePages,restore_qwen_causal_ground
from experiments.model_fabric.qualified_range_cache import QualifiedRangeCache
nav,*_=qwen_fixture()
def offline(req):raise RuntimeError('unexpected network read')
c=QualifiedRangeCache(sys.argv[2]+'/weights',offline,approved_hosts={'fixture-host'},expected_head=sys.argv[3],budget_bytes=262144)
nav.host=c
b=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
checkpoint=json.loads(sys.argv[4])
restore_qwen_causal_ground(b,LocalStatePages(sys.argv[2]+'/state'),checkpoint,expected_digest=checkpoint['checkpoint_digest'],expected_input_sequence_digest='d'*64)
output,receipt=b.next([.7,-.2,.9])
print(json.dumps({'output':output,'position':receipt['position'],'remote':c.remote_ranges,'hits':c.hits}))
'''
    proc=subprocess.run([sys.executable,'-c',script,str(src),str(tmp_path),head,json.dumps(saved)],
                        env={**__import__('os').environ,'PYTHONPATH':str(src)},capture_output=True,text=True,timeout=20)
    assert proc.returncode == 0,proc.stderr
    result=json.loads(proc.stdout)
    assert result['output']==pytest.approx(next_expected,abs=1e-12)
    assert result['position']==1
    assert result['remote']==0 and result['hits']>0
