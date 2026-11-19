from experiments.model_fabric import operational_model_lab as lab
from experiments.model_fabric.byte_lm import fit_byte_lm

def test_trained_byte_language_live_lab(tmp_path):
    corpus=tmp_path/'corpus.txt';corpus.write_bytes(b'Virtual Layer model host ' * 40)
    result=lab.run(backend='byte',prompt='Virtual Layer',steps=6,training_source=str(corpus))
    assert result['status']=='ACTUAL_TRAINED_BYTE_LM_HOST_ORBIT_VERIFIED'
    assert result['remote_weight_bytes_received']==0
    assert len(result['generated_tokens'])==6
    assert result['generated_tokens_per_s']>0
    assert len(result['checkpoint_head'])==64
    assert 'sparse_weights' not in str(result['operational_ground'])

def test_missing_transformer_backend_exits_unresolved(monkeypatch,tmp_path):
    def broken(*a,**k):raise ImportError('no torch model available')
    monkeypatch.setattr(lab,'HostedTransformersLanguage',broken)
    out=tmp_path/'report.json'
    assert lab.main(['--backend','transformer','--out',str(out)])==1
    assert 'UNRESOLVED' in out.read_text()
