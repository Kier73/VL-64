"""Fail-closed policy for the optional genuine pretrained Glance replay."""
import json
import pytest
from experiments.model_fabric import glance_model_lab


def test_only_explicitly_pinned_families_are_admitted():
    with pytest.raises(ValueError,match='pinned'):
        glance_model_lab.run('unmapped-model',root='unused')


def test_network_failure_is_unresolved_not_verified(tmp_path,monkeypatch):
    def unavailable(*args,**kwargs):
        raise TimeoutError('pinned shard source unavailable')
    monkeypatch.setattr(glance_model_lab,'_fetch_metadata',unavailable)
    result=tmp_path/'result.json'
    assert glance_model_lab.main(['--model','qwen','--root',str(tmp_path/'cache'),
                                  '--out',str(result)])==1
    report=json.loads(result.read_text())
    assert report['status']=='UNRESOLVED'
    assert report['error_type']=='TimeoutError'
    assert 'no pretrained replay verified' in report['scope']
