"""Evidence-bound portable weight Grounds: ranged reads, causal numerics, recovery."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture, expert_fixture
from test_trained_subgraphs import independent_qwen_step
from experiments.model_fabric.qualified_range_cache import QualifiedRangeCache
from experiments.model_fabric.weight_glance_ground import (
    freeze_weight_glances,PinnedWeightGlanceGround)
from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer,LocalStatePages,export_qwen_causal_ground,
    restore_qwen_causal_ground)
from experiments.model_fabric.qualified_subgraphs import observed_values

BUDGET=512*1024
LOCAL='pinned-glance-reconstructor'


def _capture(nav,root,reads):
    c=QualifiedRangeCache(root/'cache',nav.host,approved_hosts={'fixture-host'},
                          budget_bytes=BUDGET)
    nav.host=c
    for name,first,count in reads:
        nav.get_raw(name,flat_start=first,count=count)
    head=freeze_weight_glances(root/'cache',expected_cache_head=c.snapshot(),
                               ground_root=root/'ground',
                               approved_source_hosts={'fixture-host'})
    return c,head


def _replay(nav,root,head):
    g=PinnedWeightGlanceGround(root/'ground',expected_head=head,
                              approved_source_hosts={'fixture-host'})
    nav.host=g
    nav.approved={LOCAL}
    return g


def test_untouched_source_ranges_repartition_across_original_64k_boundary(tmp_path):
    raw=bytes((i*13)%256 for i in range(70000))
    nav=nav_for({'w':('I8',[70000],raw)})
    cache,head=_capture(nav,tmp_path,[('w',0,70000)])
    other=nav_for({'w':('I8',[70000],raw)})
    ground=_replay(other,tmp_path,head)
    assert other.get_raw('w',flat_start=24,count=91)[0]==raw[24:115]
    assert other.get_raw('w',flat_start=65530,count=31)[0]==raw[65530:65561]
    assert other.get_raw('w')[0]==raw
    assert ground.stats()['remote_ranges']==0
    assert cache.remote_ranges>0
    req={'model':other.index['model'],'revision':other.index['revision'],
         'index_ground_digest':other.index['ground_digest'],
         'shard':next(iter(other.weight_map.values())),
         'byte_begin':0,'byte_end_exclusive':8,'purpose':'header'}
    proof=ground(req)
    assert proof['observer_host']==LOCAL
    assert proof['glance_ground_head']==head
    assert proof['original_witnesses'][0]['observer_host']=='fixture-host'
    assert len(other.chain)==64


def test_incomplete_glances_fail_closed_no_external_network(tmp_path):
    data=b'G'*90000
    nav=nav_for({'w':('I8',[90000],data)})
    _,head=_capture(nav,tmp_path,[('w',0,8192)])
    other=nav_for({'w':('I8',[90000],data)})
    g=_replay(other,tmp_path,head)
    with pytest.raises(ValueError,match='UNRESOLVED'):
        other.get_raw('w',flat_start=8180,count=64)
    assert g.stats()['remote_ranges']==0


def test_external_head_and_historical_host_approval_required(tmp_path):
    nav=nav_for({'w':('I8',[8],b'ABCDEFGH')})
    _,head=_capture(nav,tmp_path,[('w',0,8)])
    with pytest.raises(ValueError,match='external'):
        PinnedWeightGlanceGround(tmp_path/'ground',expected_head=None,
                                 approved_source_hosts={'fixture-host'})
    with pytest.raises(ValueError,match='differs'):
        PinnedWeightGlanceGround(tmp_path/'ground',expected_head='0'*64,
                                 approved_source_hosts={'fixture-host'})
    with pytest.raises(ValueError,match='authorized'):
        PinnedWeightGlanceGround(tmp_path/'ground',expected_head=head,
                                 approved_source_hosts={'another-host'})


def test_changed_source_content_is_refused_after_freeze(tmp_path):
    nav=nav_for({'w':('I8',[8],b'ABCDEFGH')})
    _,head=_capture(nav,tmp_path,[('w',0,8)])
    page=tmp_path/'ground'/'pages'/hashlib.sha256(b'ABCDEFGH').hexdigest()
    page.write_bytes(b'BADBYTES')
    other=nav_for({'w':('I8',[8],b'ABCDEFGH')})
    _replay(other,tmp_path,head)
    with pytest.raises(ValueError,match='corrupted'):
        other.get_raw('w')


def test_overlapping_conflicting_source_witnesses_are_not_merged(tmp_path):
    nav=nav_for({'w':('I8',[100],b'A'*100)})
    original=nav.host
    def conflict(req):
        reply=original(req)
        if req['purpose']=='tensor' and req['byte_end_exclusive']-req['byte_begin']==20:
            reply['payload']=b'B'*20
            reply['sha256']=hashlib.sha256(b'B'*20).hexdigest()
        return reply
    cache=QualifiedRangeCache(tmp_path/'cache',conflict,
                              approved_hosts={'fixture-host'},budget_bytes=BUDGET)
    nav.host=cache
    nav.get_raw('w',flat_start=0,count=60)
    nav.get_raw('w',flat_start=40,count=20)
    head=freeze_weight_glances(tmp_path/'cache',expected_cache_head=cache.snapshot(),
                               ground_root=tmp_path/'ground',
                               approved_source_hosts={'fixture-host'})
    another=nav_for({'w':('I8',[100],b'A'*100)})
    _replay(another,tmp_path,head)
    with pytest.raises(ValueError,match='Contradictory'):
        another.get_raw('w',flat_start=35,count=30)


def test_qwen_three_positions_replayed_from_ground_match_independent_oracle(tmp_path):
    nav,matrices,vectors,conv=qwen_fixture()
    cache=QualifiedRangeCache(tmp_path/'cache',nav.host,
                              approved_hosts={'fixture-host'},budget_bytes=BUDGET)
    nav.host=cache
    first=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    inputs=([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])
    first.next(list(inputs[0]))
    head=freeze_weight_glances(tmp_path/'cache',expected_cache_head=cache.snapshot(),
                               ground_root=tmp_path/'ground',
                               approved_source_hosts={'fixture-host'})
    other,*_=qwen_fixture()
    g=_replay(other,tmp_path,head)
    runner=QwenQualifiedLayer(other,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    prev_conv=prev_state=None
    for x in inputs:
        result,_=runner.next(list(x))
        expected,qkv,prev_state=independent_qwen_step(
            x,matrices,vectors,conv,prev_conv,prev_state)
        prev_conv=qkv
        assert result==pytest.approx(expected,abs=2e-7)
    assert g.stats()['remote_ranges']==0


def test_new_process_restores_causal_state_and_retained_weight_ground(tmp_path):
    nav,*_=qwen_fixture()
    cache=QualifiedRangeCache(tmp_path/'cache',nav.host,
                              approved_hosts={'fixture-host'},budget_bytes=BUDGET)
    nav.host=cache
    first=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
    first.next([.1,.2,.3])
    saved=export_qwen_causal_ground(
        first,LocalStatePages(tmp_path/'state'),input_sequence_digest='d'*64)
    expected,_=first.next([.7,-.2,.9])
    head=freeze_weight_glances(tmp_path/'cache',expected_cache_head=cache.snapshot(),
                               ground_root=tmp_path/'ground',
                               approved_source_hosts={'fixture-host'})
    code="""
import sys,json
sys.path.insert(0,sys.argv[1]+'/tests')
from test_complete_native_layers import qwen_fixture
from experiments.model_fabric.weight_glance_ground import PinnedWeightGlanceGround
from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer,LocalStatePages,restore_qwen_causal_ground
nav,*_=qwen_fixture()
nav.host=PinnedWeightGlanceGround(sys.argv[2]+'/ground',expected_head=sys.argv[3],approved_source_hosts={'fixture-host'})
nav.approved={'pinned-glance-reconstructor'}
b=QwenQualifiedLayer(nav,layer=0,key_heads=1,value_heads=2,key_dim=2,value_dim=2)
state=json.loads(sys.argv[4])
restore_qwen_causal_ground(b,LocalStatePages(sys.argv[2]+'/state'),state,
    expected_digest=state['checkpoint_digest'],expected_input_sequence_digest='d'*64)
output,r=b.next([.7,-.2,.9])
print(json.dumps({'output':output,'position':r['position'],'reads':nav.host.reads}))
"""
    src=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,'-c',code,str(src),str(tmp_path),head,json.dumps(saved)],
                        env={**os.environ,'PYTHONPATH':str(src)},
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    actual=json.loads(proc.stdout)
    assert actual['output']==pytest.approx(expected,abs=1e-12)
    assert actual['position']==1 and actual['reads']>0


def test_deepseek_selected_and_shared_weights_can_replay_offline(tmp_path):
    nav,hidden=expert_fixture(shared_dtype='F8_E4M3')
    cache=QualifiedRangeCache(tmp_path/'cache',nav.host,
                              approved_hosts={'fixture-host'},budget_bytes=BUDGET)
    nav.host=cache
    from experiments.model_fabric.complete_native_layers import DeepSeekQualifiedMoE
    first=DeepSeekQualifiedMoE(nav,layer=0)
    expected,r=first.next(hidden)
    head=freeze_weight_glances(tmp_path/'cache',expected_cache_head=cache.snapshot(),
                               ground_root=tmp_path/'ground',
                               approved_source_hosts={'fixture-host'})
    other,_=expert_fixture(shared_dtype='F8_E4M3')
    g=_replay(other,tmp_path,head)
    got,receipt=DeepSeekQualifiedMoE(other,layer=0).next(hidden)
    assert got==pytest.approx(expected,abs=1e-10)
    assert receipt['selected_experts']==r['selected_experts']==list(range(6))
    assert g.stats()['remote_ranges']==0
