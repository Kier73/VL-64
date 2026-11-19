"""Archived actual pretrained model-host tokens observed through VL64's operational Ground.

Scope: verify exact Actions report bytes, model pin, operator Ground, and accounting.
Does not independently rerun the pretrained transformer mathematics or assert
that the tiny base model provides instruction-quality text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.model_fabric.operational_ground import (
    GROUND_SCHEMA, digest,
)

RECORD = (Path(__file__).resolve().parents[1] / 'experiments' / 'model_fabric' /
          'datasets' / '2026-11-19-pretrained-operational-model-host.json')


def test_archived_pinned_transformer_host_report_exact_artifact():
    raw = RECORD.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        'b45ac96c1068ce67e2d97adf08f6f08a9582d2e14526026d53cc13d9c2241deb')
    report = json.loads(raw)
    assert report['schema'] == 'vl64.operational-language-benchmark.v1'
    assert report['status'] == 'PRETRAINED_TRANSFORMER_HOST_ORBIT_OBSERVED'
    assert report['model'] == 'HuggingFaceTB/SmolLM2-135M'
    assert report['revision'] == '4e53f736cbb20a9a0f56b4c4bf378d9f306ff915'
    assert report['backend'] == 'transformer'
    assert report['prompt'] == 'The capital of France is'
    assert report['generated_tokens'] == [260, 3575, 282]
    assert report['generated_text'] == ' the capital of'
    assert report['host_calls'] == 2 + 3 * (12 + 1) + 1 == 42
    assert report['remote_weight_bytes_received'] == 0
    assert report['reference'] == 'complete_host_logit_tiles_not_independent_transformer_math'
    assert len(report['checkpoint_head']) == 64
    assert len(report['transcript_head']) == 64


def test_pinned_operational_ground_metadata_integrity_without_weight_attestation():
    report = json.loads(RECORD.read_bytes())
    ground = report['operational_ground']
    assert ground['schema'] == GROUND_SCHEMA
    assert ground['ground_digest'] == digest(GROUND_SCHEMA, {
        key:value for key,value in ground.items() if key != 'ground_digest'})
    assert ground['model'] == report['model']
    assert ground['revision'] == report['revision']
    assert ground['vocab_size'] == 49152
    assert ground['max_context'] == 256
    assert ground['operations'] == ['encode','decode','open','logits_tile','advance']
    assert ground['observer_host'] == 'torch-model-host'
    assert ground['trust_scope'].endswith('not_remote_weight_or_execution_attestation')
    assert ground['model_artifact_identity'] == digest(
        'vl64.host-model-revision-claim.v1',
        {'model':ground['model'],'revision':ground['revision']})
    assert 'weights' not in ground and 'tensor_bytes' not in ground
