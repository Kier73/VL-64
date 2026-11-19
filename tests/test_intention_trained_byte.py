"""Real trained byte-LM operation through typed intent + host-resident parameters."""
import json
from pathlib import Path

from experiments.model_fabric.byte_lm import ByteLanguageGround
from experiments.model_fabric.operational_ground import observe_operational_ground
from experiments.model_fabric.operational_hosts import HostedTrainedByteLanguage
from experiments.model_fabric.intention_fabric import compile_intention,IntentionSession

GROUND_PATH=Path(__file__).resolve().parents[1]/'experiments/model_fabric/datasets/2026-11-18-trained-byte-language-ground.json'


def test_actual_trained_byte_model_intent_and_token_oracle():
    weights=json.loads(GROUND_PATH.read_text(encoding='utf-8'))
    reference=ByteLanguageGround(weights)
    host=HostedTrainedByteLanguage(weights,observer_host='trained-byte-host')
    ground=observe_operational_ground(host,expected_model=host.description['model'],
         expected_revision=host.description['revision'],approved_hosts={'trained-byte-host'},
         expected_artifact=weights['ground_digest'])
    intention=compile_intention(ground,kind='paragraph',messages=[
         {'role':'user','content':'Explain the virtual ground.'}],max_new_tokens=12)
    nav=IntentionSession(intention,ground,host,approved_hosts={'trained-byte-host'})
    out=nav.run(tile_size=73)
    actual=out['generated_tokens']
    original_prefix=nav.nav.tokens[:-len(actual)]
    prev=original_prefix[-1]
    oracle=[]
    for _ in actual:
        nxt=reference.predict(prev,limit=1)['ranked_tokens'][0]['byte']
        oracle.append(nxt);prev=nxt
    assert actual==oracle
    assert out['remote_weight_bytes_received']==0
    assert out['status'] in {'OUTPUT_SHAPE_ADMITTED','OUTPUT_CONTRACT_UNRESOLVED'}
    assert len(out['transcript_head'])==64
    assert 'sparse_weights' not in json.dumps(intention)
