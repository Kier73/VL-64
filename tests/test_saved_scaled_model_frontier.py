"""Regression for exact successful pinned large-model Actions observation metadata."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.model_fabric.scaled_model_lab import PINNED

ROOT=Path(__file__).resolve().parents[1]
FILE=(ROOT/'experiments/model_fabric/datasets/'
      '2026-11-19-large-model-frontier-observations.json')


def test_real_large_model_observation_claims_are_pinned_and_scoped():
    evidence=json.loads(FILE.read_text(encoding='utf-8'))
    assert evidence['schema']=='vl64.architecture-scale-observations.v1'
    assert set(evidence['models'])=={'qwen','deepseek'}
    expected={'qwen':(64,18,1199,10240),
              'deepseek':(40,48,96085,1024)}
    for name,meta in evidence['models'].items():
        m=meta['report']
        layers,shards,tensors,bytes_observed=expected[name]
        assert m['model']==PINNED[name][0]
        assert m['revision']==PINNED[name][1]
        assert m['layers']==layers
        assert m['shards']==shards
        assert m['mapped_tensors']==tensors
        assert m['pretrained_weight_bytes_observed']==bytes_observed
        assert m['numeric_probe']['actual_weight_bytes']==bytes_observed
        assert m['numeric_probe']['max_abs_error']<=1e-7
        assert m['numeric_probe']['status']=='WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED'
        assert m['status']=='STRUCTURAL_MODEL_GROUND_ADMITTED'
        assert m['full_model_executed'] is False
        assert m['exact_http_ranges']>=3
        assert len(m['range_witness_chain_digest'])==64
        assert len(m['ground_digest'])==64
        assert meta['run_id']==44541524537
