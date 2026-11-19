"""Independent trained byte models, not scripted generation, in two VL instances."""
from experiments.model_fabric.intention_fleet_lab import byte_fleet,main


def test_real_two_training_ground_isolated_fleet(tmp_path):
    data=(b'local observer ground and tensor operation 0123456789\n'*30+
          b'parsing language and model calculation abcdefghijklmnopqrstuv\n'*30)
    corpus=tmp_path/'training.txt';corpus.write_bytes(data)
    out=byte_fleet(str(corpus),steps=6)
    assert out['status']=='TWO_INDEPENDENT_TRAINED_MODEL_HOSTS_EXECUTED'
    assert len(out['instances'])==2
    assert out['instances'][0]['model_training_digest']!=out['instances'][1]['model_training_digest']
    assert all(x['num_generated_tokens']==6 and x['transferred_weight_bytes']==0 and
               x['independent_trained_weight_oracle_match'] for x in out['instances'])
    dest=tmp_path/'out.json'
    assert main(['--training-source',str(corpus),'--steps','3','--out',str(dest)])==0
    assert dest.exists()
