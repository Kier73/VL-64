"""Host-resident model generation and untrusted-processor refusal tests."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from experiments.model_fabric.byte_lm import ByteLanguageGround, fit_byte_lm
from experiments.model_fabric.operational_ground import (
    OperationalModelNavigator, BoundedTaskAgent, observe_operational_ground,
    digest, GROUND_SCHEMA, RESULT_SCHEMA, prefix_digest,
)
from experiments.model_fabric.operational_hosts import (
    HostedTrainedByteLanguage, OperationalHostBase,
)

CORPUS=(b'Virtual Layer Dynamics provides a virtual layer for computation. '
        b'The observer-host performs trained operations. The ground navigates results. ')*15


def make_model(host_id='local-byte-model-host'):
    training=fit_byte_lm(CORPUS)
    host=HostedTrainedByteLanguage(training,observer_host=host_id)
    g=observe_operational_ground(host,expected_model='vl64/trained-byte-language',
            expected_revision=training['ground_digest'],approved_hosts={host_id},
            expected_artifact=training['ground_digest'])
    return host,g,training


def test_trained_text_processing_uses_host_weights_without_weight_copies():
    host,g,training=make_model()
    nav=OperationalModelNavigator(g,host,approved_hosts={g['observer_host']})
    out=nav.generate('Virtual Layer',max_new_tokens=12,tile_size=29)
    oracle=ByteLanguageGround(training).generate(b'Virtual Layer',steps=12)
    assert bytes(out['generated_tokens']).hex()==oracle['generated_hex']
    assert out['generated_text']==bytes(out['generated_tokens']).decode('utf-8','replace')
    assert out['status']=='HOST_RESIDENT_MODEL_TOKENS_EXECUTED'
    assert out['remote_weight_bytes_received']==0
    assert 'sparse_weights' not in json.dumps(g)
    assert len(json.dumps(g))<1024
    assert nav.remote_calls>=12*9 and len(out['position_receipts'])==12


def test_metadata_is_not_weight_attestation():
    host,g,_=make_model()
    assert 'not_remote_weight_or_execution_attestation' in g['trust_scope']
    with pytest.raises(ValueError,match='artifact'):
        observe_operational_ground(host,expected_model=g['model'],
           expected_revision=g['revision'],approved_hosts={g['observer_host']},
           expected_artifact='0'*64)
    forged=dict(g);forged['model_artifact_identity']='0'*64
    with pytest.raises(ValueError):
        OperationalModelNavigator(forged,host,approved_hosts={g['observer_host']})


def test_wrong_observer_not_admitted():
    host,g,_=make_model()
    with pytest.raises(ValueError,match='Unapproved'):
        OperationalModelNavigator(g,host,approved_hosts={'another-host'})
    with pytest.raises(ValueError,match='Host model'):
        observe_operational_ground(host,expected_model=g['model'],
                    expected_revision='unrelated',approved_hosts={g['observer_host']})


@pytest.mark.parametrize('tile_size',[1,7,129,256])
def test_all_logit_tiles_choose_the_same_token(tile_size):
    host,g,_=make_model()
    nav=OperationalModelNavigator(g,host,approved_hosts={g['observer_host']})
    nav.open_text('Virtual')
    token=nav.project_next_token(tile_size=tile_size)['token']
    assert token==nav.project_next_token(tile_size=256)['token']


def test_misaddressed_tile_and_overwritten_tile_digest_refused():
    host,g,_=make_model()
    def bad(req):
        reply=host(req)
        if req['operation']=='logits_tile':
            reply=copy.deepcopy(reply)
            reply['result']['start']+=1
            reply['result_digest']=digest(RESULT_SCHEMA,reply['result'])
        return reply
    nav=OperationalModelNavigator(g,bad,approved_hosts={g['observer_host']})
    nav.open_text('Virtual')
    with pytest.raises(ValueError,match='tile'):nav.advance()
    assert nav.tokens==list(b'Virtual')


def test_host_full_logit_commitment_prevents_self_consistent_tile_replacement():
    host,g,_=make_model()
    def bad(req):
        reply=host(req)
        if req['operation']=='logits_tile':
            reply=copy.deepcopy(reply)
            raw=bytearray.fromhex(reply['result']['f32le_hex']);raw[0]^=1
            reply['result']['f32le_hex']=raw.hex()
            reply['result']['sha256']=hashlib.sha256(raw).hexdigest()
            reply['result_digest']=digest(RESULT_SCHEMA,reply['result'])
        return reply
    nav=OperationalModelNavigator(g,bad,approved_hosts={g['observer_host']})
    nav.open_text('Virtual')
    with pytest.raises(ValueError,match='commitment'):nav.advance()
    assert nav.tokens==list(b'Virtual')


def test_missing_tile_or_nonfinite_logit_refused():
    host,g,_=make_model()
    def bad(req):
        reply=host(req)
        if req['operation']=='logits_tile':
            reply=copy.deepcopy(reply)
            raw=bytearray.fromhex(reply['result']['f32le_hex'])
            raw[:4]=struct.pack('<f',float('nan'))
            reply['result']['f32le_hex']=raw.hex()
            reply['result']['sha256']=hashlib.sha256(raw).hexdigest()
            reply['result_digest']=digest(RESULT_SCHEMA,reply['result'])
        return reply
    nav=OperationalModelNavigator(g,bad,approved_hosts={g['observer_host']})
    nav.open_text('V')
    with pytest.raises(ValueError,match='Nonfinite'):nav.advance()


def test_checkpoint_recovered_on_new_host_with_exact_logit_identity():
    host,g,training=make_model()
    one=OperationalModelNavigator(g,host,approved_hosts={g['observer_host']})
    one.open_text('Layer')
    one.advance(tile_size=23)
    checkpoint=one.checkpoint()
    other=HostedTrainedByteLanguage(training,observer_host=g['observer_host'])
    restored=OperationalModelNavigator.restore(g,other,
         approved_hosts={g['observer_host']},checkpoint=checkpoint,
         expected_checkpoint_digest=checkpoint['checkpoint_digest'],owner='replacement-process',epoch=1)
    assert restored.tokens==one.tokens and restored.logits_digest==one.logits_digest
    assert restored.advance(tile_size=12)==one.advance(tile_size=24)
    assert restored.transcript_head!=one.transcript_head  # tile partitioned differently
    assert restored.remote_weight_bytes_received==0


def test_checkpoint_stale_and_altered_refused():
    host,g,training=make_model()
    nav=OperationalModelNavigator(g,host,approved_hosts={g['observer_host']})
    nav.open_text('Layer');nav.advance()
    checkpoint=nav.checkpoint()
    kwargs=dict(ground=g,host=HostedTrainedByteLanguage(training),
             approved_hosts={g['observer_host']},checkpoint=checkpoint,
             expected_checkpoint_digest=checkpoint['checkpoint_digest'],owner='new',epoch=1)
    with pytest.raises(ValueError):OperationalModelNavigator.restore(**dict(kwargs,epoch=0))
    with pytest.raises(ValueError):OperationalModelNavigator.restore(**dict(kwargs,expected_checkpoint_digest='0'*64))
    changed=dict(checkpoint);changed['tokens']=checkpoint['tokens']+[1]
    with pytest.raises(ValueError):OperationalModelNavigator.restore(**dict(kwargs,checkpoint=changed))


def test_actual_cross_process_restart(tmp_path):
    host,g,training=make_model()
    nav=OperationalModelNavigator(g,host,approved_hosts={g['observer_host']})
    nav.open_text('Hello');nav.advance()
    checkpoint=nav.checkpoint()
    values=tmp_path/'transfer.json';values.write_text(json.dumps([g,training,checkpoint]))
    script='''
import json,sys
from experiments.model_fabric.operational_ground import OperationalModelNavigator
from experiments.model_fabric.operational_hosts import HostedTrainedByteLanguage
with open(sys.argv[1]) as f:g,trained,state=json.load(f)
host=HostedTrainedByteLanguage(trained,observer_host=g['observer_host'])
nav=OperationalModelNavigator.restore(g,host,approved_hosts={g['observer_host']},checkpoint=state,
   expected_checkpoint_digest=state['checkpoint_digest'],owner='child',epoch=1)
token=nav.advance()
print(json.dumps({'token':token,'head':nav.checkpoint()['checkpoint_digest'],'received_weights':nav.remote_weight_bytes_received}))
'''
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    sub=subprocess.run([sys.executable,'-c',script,str(values)],env=env,capture_output=True,text=True,check=True)
    result=json.loads(sub.stdout)
    assert result['token']==nav.advance()
    assert result['received_weights']==0 and len(result['head'])==64


class WideSyntheticHost(OperationalHostBase):
    def __init__(self,vocab=49152):
        super().__init__(model='synthetic/wide-model',revision='a'*40,
            observer_host='wide-host',artifact_identity='f'*64,
            tokenizer_id='synthetic_byte_v1',vocab_size=vocab,max_context=128)
    def encode(self,text):return [ord(c)%self.description['vocab_size'] for c in text]
    def decode(self,tokens):return ','.join(map(str,tokens))
    def calculate_logits(self,tokens):
        return [-abs(i-33333)/100 for i in range(self.description['vocab_size'])]


def test_large_vocabulary_tiles_have_bounded_frame_and_complete_argmax():
    host=WideSyntheticHost()
    g=observe_operational_ground(host,expected_model=host.description['model'],
       expected_revision=host.description['revision'],approved_hosts={'wide-host'})
    nav=OperationalModelNavigator(g,host,approved_hosts={'wide-host'})
    nav.open_text('A')
    projected=nav.project_next_token(tile_size=4096)
    assert projected['token']==33333 and projected['logit_tiles']==12
    assert nav.advance()==33333
    assert nav.remote_weight_bytes_received==0


def test_larger_than_bounded_tile_refused():
    host=WideSyntheticHost()
    g=observe_operational_ground(host,expected_model=host.description['model'],
       expected_revision=host.description['revision'],approved_hosts={'wide-host'})
    nav=OperationalModelNavigator(g,host,approved_hosts={'wide-host'})
    nav.open_text('A')
    with pytest.raises(ValueError,match='tile'):nav.advance(tile_size=10000)


def test_bounded_agent_refuses_unknown_tool_and_runs_allowed_tool():
    def factory():return object()
    class Scripted:
        def __init__(self,text):self.text=text
        def generate(self,*a,**kw):return {'generated_text':self.text}
    bad=BoundedTaskAgent(lambda: Scripted('{"tool":"shell","arguments":{"command":"rm"}}'),
                         tools={'safe_add':lambda a,b:a+b})
    with pytest.raises(ValueError,match='unapproved'):bad.run('add numbers')
    calls=iter(['{"tool":"safe_add","arguments":{"a":3,"b":4}}','{"final":"7"}'])
    good=BoundedTaskAgent(lambda: Scripted(next(calls)),
                          tools={'safe_add':lambda a,b:a+b})
    result=good.run('add 3 and 4')
    assert result['status']=='BOUNDED_MODEL_TASK_COMPLETED'
    assert result['final']=='7' and len(result['events'])==1


def test_agent_refuses_malformed_decision_and_limits_rounds():
    class Scripted:
        def generate(self,*a,**kw):return {'generated_text':'not JSON'}
    agent=BoundedTaskAgent(lambda:Scripted(),tools={})
    with pytest.raises(ValueError,match='structured'):agent.run('What is this?')


def test_reassignment_to_different_qualified_host_observes_same_causal_logits():
    host,g,training=make_model('primary-operator-host')
    nav=OperationalModelNavigator(g,host,approved_hosts={'primary-operator-host'})
    nav.open_text('Virtual'); nav.advance()
    checkpoint=nav.checkpoint()
    replacement=HostedTrainedByteLanguage(training,observer_host='replacement-operator-host')
    migrated,evidence=nav.transfer_to(replacement,
        new_approved_hosts={'replacement-operator-host'},
        expected_checkpoint_digest=checkpoint['checkpoint_digest'],owner='replacement',epoch=1)
    assert migrated.tokens==nav.tokens
    assert migrated.ground['ground_digest']!=g['ground_digest']
    assert evidence['source_ground']==g['ground_digest']
    assert evidence['replacement_ground']==migrated.ground['ground_digest']
    assert migrated.advance()==nav.advance()
    assert migrated.remote_weight_bytes_received==0


def test_reassignment_requires_external_checkpoint_pin_and_same_model_identity():
    host,g,training=make_model('primary-operator-host')
    nav=OperationalModelNavigator(g,host,approved_hosts={'primary-operator-host'})
    nav.open_text('Virtual')
    new=HostedTrainedByteLanguage(training,observer_host='replacement-operator-host')
    with pytest.raises(ValueError,match='high-water'):
        nav.transfer_to(new,new_approved_hosts={'replacement-operator-host'},
            expected_checkpoint_digest='0'*64,owner='other',epoch=1)
    bad_training=fit_byte_lm(CORPUS+b' changed')
    changed=HostedTrainedByteLanguage(bad_training,observer_host='replacement-operator-host')
    with pytest.raises(ValueError,match='artifact'):
        nav.transfer_to(changed,new_approved_hosts={'replacement-operator-host'},
            expected_checkpoint_digest=nav.checkpoint()['checkpoint_digest'],owner='other',epoch=1)
