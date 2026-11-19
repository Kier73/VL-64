"""Optional live pretrained retention harness: explicit failure, no network required."""
from __future__ import annotations

import json
from experiments.model_fabric import retained_model_lab as lab


def test_missing_pinned_model_refused(tmp_path):
    try:lab.run('unknown',cache_dir=tmp_path)
    except ValueError as e:assert 'pinned' in str(e)
    else:raise AssertionError('Unknown model admitted')


def test_unreachable_pinned_checkpoint_leaves_unresolved_receipt(tmp_path,monkeypatch):
    def fail(*args, **kwargs):raise TimeoutError('source unreachable')
    monkeypatch.setattr(lab, '_fetch_metadata', fail)
    output = tmp_path / 'out.json'
    assert lab.main(['--model','qwen','--cache-dir',str(tmp_path/'cache'),
                     '--out',str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'UNRESOLVED'
    assert report['error_type'] == 'TimeoutError'


def test_repeat_probe_rejects_falsely_claimed_cache_hit(tmp_path,monkeypatch):
    import copy
    from test_architecture_probe import nav_for
    from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
    # Use local fake metadata only to check the refusal contract. The named
    # pinned checkpoint's real numeric evidence still requires remote bytes.
    monkeypatch.setattr(lab,'_fetch_metadata',lambda *x: b'{"weight_map":{"w":"weights.safetensors"}}')
    def no_weights(*args, **kwargs):
        raise ValueError('Unadmitted tensor weight source')
    monkeypatch.setattr(lab,'pinned_hf_range_host', no_weights)
    try:lab.run('qwen',cache_dir=tmp_path)
    except (ValueError,TypeError) as e:assert 'Unadmitted' in str(e) or 'missing' in str(e) or 'tensor' in str(e)
    else:raise AssertionError('Missing actual pinned weight ranges accepted')
