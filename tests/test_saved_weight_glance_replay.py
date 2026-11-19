"""Re-check permanently archived actual pretrained Ground replay evidence offline."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'experiments/model_fabric/datasets'
SHA256=re.compile(r'[0-9a-f]{64}\Z')


def test_archived_real_pretrained_weight_glance_replay_is_scoped_and_complete():
    record=json.loads((ROOT/'2026-11-19-real-trained-weight-glance-replays.json').read_text())
    assert record['schema']=='vl64.real-pretrained-glance-replay-archive.v1'
    assert record['github_actions_run_id']==44585746575
    assert set(record['observations'])=={'qwen','deepseek'}
    expected={
        'qwen':('Qwen/Qwen3.8-27B',100,5322588,101,10),
        'deepseek':('deepseek-ai/DeepSeek-V4.1-Flash-1017',106,4231824,107,20),
    }
    for family,(name,remote_count,transfer,replay_count,short_bytes) in expected.items():
        r=record['observations'][family]
        assert r['schema']=='vl64.real-pretrained-glance-replay.v1'
        assert r['status']=='REAL_PINNED_GLANCE_NUMERICAL_REPLAY_VERIFIED'
        assert r['model']==name
        assert re.fullmatch(r'[0-9a-f]{40}',r['revision'])
        assert r['original_remote_ranges']==remote_count
        assert r['original_remote_bytes']==transfer
        assert r['replayed_remote_ranges']==0
        assert r['replayed_local_ranges']==replay_count
        assert r['repartitioned_weight_span_bytes']==short_bytes
        assert r['replayed_local_bytes']==transfer+short_bytes
        assert r['original_observation']==r['retained_replay']
        assert r['original_observation']['reference_error']==0.0
        for field in ('index_ground_digest','source_catalogue_head',
                      'portable_ground_head','replay_range_chain'):
            assert SHA256.fullmatch(r[field])
        assert 'not a complete large-model layer' in r['scope']
    q=record['observations']['qwen']['original_observation']
    assert q['status']=='REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED'
    assert q['output']==-0.003032839242
    d=record['observations']['deepseek']['original_observation']
    assert d['status']=='REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED'
    assert d['selected_experts']==[240,360,2,34,164,199]
    assert len(d['routing_weights'])==6
