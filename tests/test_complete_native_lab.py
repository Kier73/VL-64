"""No-network live-harness refusal and evidence-scope regression."""
from experiments.model_fabric import complete_native_lab as lab


def test_unknown_source_cannot_masquerade_as_complete_layer():
    try:lab.run('unmapped')
    except KeyError:pass
    else:raise AssertionError('Unknown model was accepted')


def test_cli_fails_closed_when_a_pinned_source_is_unreachable(monkeypatch,tmp_path):
    def fail(*args,**kwargs):raise TimeoutError('unreachable host')
    monkeypatch.setattr(lab,'_fetch_metadata',fail)
    out=tmp_path/'report.json'
    assert lab.main(['--model','qwen','--execute','--out',str(out)])==1
    import json
    report=json.loads(out.read_text())
    assert report['status']=='UNRESOLVED' and report['error_type']=='TimeoutError'
    assert 'No pretrained' in report['scope']
