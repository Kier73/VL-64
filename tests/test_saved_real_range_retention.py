"""Reconcile saved real pinned Qwen/DeepSeek range reuse with source manifests.

These are *two same-input subgraph evaluations*, not complete model layers.
All checks run without HTTPS or pretrained payload downloads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.model_fabric.scaled_model_lab import PINNED

SOURCE = Path(__file__).resolve().parents[1] / (
    'experiments/model_fabric/datasets/'
    '2026-11-19-real-retained-pretrained-range-observations.json'
)
HEX = re.compile(r'[0-9a-f]{64}\Z')


def test_archived_live_range_reuse_receipts():
    record = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert record['schema'] == 'vl64.pinned-weight-range-reuse-observations.v1'
    assert record['github_actions_run_id'] == 44579893703
    assert record['jobs'] == {'qwen': 135757375622, 'deepseek': 135757376153}
    expected = {
        'qwen': {'tensor_bytes': 5277188, 'header_bytes': 45400,
                 'ranges': 100, 'hits': 98},
        'deepseek': {'tensor_bytes': 3973680, 'header_bytes': 258144,
                     'ranges': 106, 'hits': 101},
    }
    for model, numbers in expected.items():
        r = record[model]
        assert (r['model'], r['revision']) == PINNED[model]
        assert r['schema'] == 'vl64.real-pretrained-range-retention.v1'
        assert r['status'] == 'REAL_PINNED_WEIGHT_REUSE_AND_REFERENCE_VERIFIED'
        assert r['repeat_count'] == 2
        assert r['remote_ranges_after_first'] == r['remote_ranges_after_second'] == numbers['ranges']
        assert r['reused_range_requests'] == numbers['hits']
        assert r['first_pass_tensor_bytes'] == numbers['tensor_bytes']
        assert r['remote_header_bytes'] == numbers['header_bytes']
        assert r['remote_transfer_bytes_total'] == numbers['tensor_bytes'] + numbers['header_bytes']
        assert r['subgraph']['reference_error'] == 0.0
        for field in ('index_ground_digest', 'original_witness_cache_head',
                      'range_transcript_digest'):
            assert HEX.fullmatch(r[field])
        assert 'not independent real full-layer inference' in r['scope']
    assert record['qwen']['subgraph']['head'] == 0
    assert record['qwen']['subgraph']['numerical_output'] == -0.003032839242
    assert record['deepseek']['subgraph']['selected_experts'] == [240, 360, 2, 34, 164, 199]
    assert record['deepseek']['subgraph']['selected_slice_receipt'] == (
        'REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED')
