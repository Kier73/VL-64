"""Independent full 30-layer logits, restart, and evidence-failure checks."""
import copy
import os
import json
import math
import tempfile
from pathlib import Path

import pytest
from vld import domain_digest
from experiments.model_fabric.execution_dag import compile_decoder_dag
from experiments.model_fabric.full_decoder import (
    StreamingDecoder, save_full_checkpoint, load_full_checkpoint,
)
from experiments.model_fabric.pretrained_orbit import ContentAddressedPages
from test_execution_dag import synthetic_pretrained_ground,qualified_host,scalar_reference


def fixture(tmp_path,*,layers=30,dtype='BF16',tied=True):
    ground,data,weights=synthetic_pretrained_ground(layers=layers,dtype=dtype,tied=tied)
    plan=compile_decoder_dag(ground)
    return plan,weights,qualified_host(ground,data),ContentAddressedPages(tmp_path/'pages')


def make(plan,host,store,tokens,**kwargs):
    return StreamingDecoder(plan,tokens,lambda index:host,
        lambda index:{'local-f32-observer'},store,**kwargs)


@pytest.mark.parametrize('dtype',['BF16','F32'])
@pytest.mark.parametrize('tied',[True,False])
def test_all_30_layers_match_independent_full_reference(tmp_path,dtype,tied):
    plan,weights,host,store=fixture(tmp_path,layers=30,dtype=dtype,tied=tied)
    tokens=[4,3,6]
    s=make(plan,host,store,tokens)
    for i in range(30):
        r=s.advance_layer()
        assert r['completed_layers']==i+1
        assert r['layer_weight_bytes']>0
        assert len(s.kv_manifests)==i+1
    receipt,logits=s.logits(return_logits=True)
    ref=scalar_reference(tokens,weights,{
        'num_hidden_layers':30,'hidden_size':8,'num_attention_heads':4,
        'num_key_value_heads':2,'intermediate_size':12,'vocab_size':11,
        'rms_norm_eps':1e-5,'rope_theta':10000.0,'tie_word_embeddings':tied})
    assert len(logits)==len(ref)==11
    assert max(abs(a-b) for a,b in zip(logits,ref))<2e-6
    assert receipt['layers_executed']==30
    assert receipt['selected_next_token']==max(range(11),key=lambda i:ref[i])
    assert receipt['weight_payload_bytes_acquired']>s.bytes_acquired
    assert receipt['exact_range_count']>s.ranges_acquired
    assert all(len(v)==64 for v in s.weight_identities.values())


def test_restart_after_layer_17_rehomes_without_replay(tmp_path):
    plan,weights,host,store=fixture(tmp_path)
    tokens=[2,8]
    direct=make(plan,host,store,tokens)
    for i in range(30):direct.advance_layer()
    expected,logits=direct.logits(return_logits=True)
    start=make(plan,host,store,tokens,owner='first',epoch=2)
    for i in range(17):start.advance_layer()
    snap=start.checkpoint()
    f=tmp_path/'checkpoint.json'
    head=save_full_checkpoint(f,snap)
    read=load_full_checkpoint(f)
    resumed=make(plan,host,store,tokens,owner='replacement',epoch=3,
                 checkpoint=read,expected_head=head)
    assert resumed.completed_layers==17
    for i in range(17,30):resumed.advance_layer()
    got,values=resumed.logits(return_logits=True)
    assert values==logits
    assert got['selected_next_token']==expected['selected_next_token']
    assert resumed.ranges_acquired==direct.ranges_acquired
    assert resumed.checkpoint()['previous_checkpoint']==head
    assert resumed.checkpoint()['epoch']==3


def test_checkpoint_requires_external_trusted_head_and_higher_epoch(tmp_path):
    plan,_,host,store=fixture(tmp_path,layers=3)
    runner=make(plan,host,store,[2],epoch=5)
    runner.advance_layer()
    snap=runner.checkpoint()
    with pytest.raises(ValueError,match='pinned|epoch'):
        make(plan,host,store,[2],checkpoint=snap,owner='other',epoch=5,
             expected_head=snap['checkpoint_digest'])
    with pytest.raises(ValueError,match='pinned|frontier'):
        make(plan,host,store,[2],checkpoint=snap,owner='other',epoch=6,
             expected_head='0'*64)
    changed=copy.deepcopy(snap)
    changed['tokens']=[3]
    changed['checkpoint_digest']=domain_digest(changed['schema'],{
        k:v for k,v in changed.items() if k!='checkpoint_digest'})
    with pytest.raises(ValueError,match='frontier'):
        make(plan,host,store,[2],checkpoint=changed,owner='other',epoch=6,
             expected_head=changed['checkpoint_digest'])


def test_corrupt_kv_page_fails_before_any_new_host_weight_read(tmp_path):
    plan,_,host,store=fixture(tmp_path,layers=4)
    runner=make(plan,host,store,[2,3]);runner.advance_layer()
    snap=runner.checkpoint()
    page=store.root/snap['kv_manifests'][0]
    page.write_bytes(b'wrong\x00')
    seen=[]
    def no_host(index):
        seen.append(index)
        return host
    with pytest.raises(ValueError,match='KV|digest|state page'):
        StreamingDecoder(plan,[2,3],no_host,lambda i:{'local-f32-observer'},
                         store,epoch=1,owner='recovery',checkpoint=snap,
                         expected_head=snap['checkpoint_digest'])
    assert seen==[]


def test_checkpoint_layer_failure_atomic(tmp_path):
    plan,_,host,store=fixture(tmp_path,layers=4)
    runner=make(plan,host,store,[3,2])
    runner.advance_layer()
    snap=runner.checkpoint()
    def bad_source(i):
        if i==1:
            def fail(request):
                return dict(host(request),sha256='0'*64)
            return fail
        return host
    runner.host_for=bad_source
    with pytest.raises(ValueError,match='Unqualified'):
        runner.advance_layer()
    assert runner.completed_layers==1
    assert runner.checkpoint()==snap
    runner.host_for=lambda i:host
    runner.advance_layer()
    assert runner.completed_layers==2


def test_unfinished_model_forbids_vocabulary_claim(tmp_path):
    plan,_,host,store=fixture(tmp_path,layers=30)
    runner=make(plan,host,store,[2])
    runner.advance_layer()
    with pytest.raises(ValueError,match='every declared layer'):
        runner.logits()


def test_final_dag_change_fails_even_if_digest_recomputed(tmp_path):
    plan,_,host,store=fixture(tmp_path,layers=1)
    plan=copy.deepcopy(plan)
    plan['nodes'][-1]['weight']='model.norm.weight'
    plan['dag_digest']=domain_digest(plan['schema'],{
        k:v for k,v in plan.items() if k!='dag_digest'})
    runner=make(plan,host,store,[2])
    runner.advance_layer()
    with pytest.raises(ValueError,match='vocabulary DAG contract'):
        runner.logits()

@pytest.mark.parametrize('dtype',['BF16','F32'])
def test_numpy_reference_for_complete_30_layer_fixture(tmp_path,dtype):
    from collections import defaultdict
    import numpy as np
    from experiments.model_fabric.full_decoder_lab import (
        independent_numpy_layer,independent_numpy_vocab)
    plan,weights,host,store=fixture(tmp_path,layers=30,dtype=dtype)
    captured=defaultdict(list)
    def collecting(req):
        response=host(req)
        captured[req['tensor']].append((req['byte_begin'],response['payload']))
        return response
    runner=make(plan,collecting,store,[4,3])
    for layer in range(30):
        before=runner.hidden
        captured.clear()
        runner.advance_layer()
        reference=independent_numpy_layer(plan,captured,layer,before)
        assert max(float(np.max(abs(np.asarray(x)-r))) for x,r in zip(runner.hidden,reference))<2e-7
    captured.clear()
    receipt,got=runner.logits(return_logits=True)
    expected=independent_numpy_vocab(plan,captured,runner.hidden[-1])
    assert float(np.max(abs(np.asarray(got)-expected)))<2e-7
    assert receipt['selected_next_token']==int(np.argmax(expected))


def test_real_lab_two_process_cli_plumbing_on_mock_full_model(tmp_path,monkeypatch):
    import numpy as np
    from experiments.model_fabric import full_decoder_lab as lab
    ground,data,weights=synthetic_pretrained_ground(layers=30,dtype='BF16')
    ground_file=tmp_path/'pinned-fixture.json'
    ground_file.write_text(json.dumps(ground))
    monkeypatch.setattr(lab,'MANIFEST',ground_file)
    base=qualified_host(ground,data)
    def remote_reader(plan):
        def host(request):
            response=base(request)
            response['observer_host']='pinned-huggingface-https-range'
            return response
        return host
    monkeypatch.setattr(lab,'huggingface_range_host',remote_reader)
    state_dir=tmp_path/'state'
    partial=lab.run(tokens=(3,4),through_layer=17,state_dir=state_dir,
                    owner='process-A',epoch=0,result_path=tmp_path/'partial.json')
    assert partial['status']=='VERIFIED_PRETRAINED_LAYER_FRONTIER'
    checkpoint=state_dir/'checkpoint.json'
    complete=lab.run(tokens=(3,4),through_layer=30,state_dir=state_dir,
                     owner='process-B',epoch=1,resume_from=checkpoint,
                     expected_head=partial['checkpoint_digest'],
                     result_path=tmp_path/'complete.json')
    assert complete['status']=='REAL_PRETRAINED_30_LAYER_LOGITS_INDEPENDENTLY_VERIFIED'
    assert complete['previous_checkpoint_digest']==partial['checkpoint_digest']
    assert complete['verified_layers_this_process']==13
    assert complete['full_vocab_size']==11
    ref=scalar_reference([3,4],weights,{
        'num_hidden_layers':30,'hidden_size':8,'num_attention_heads':4,
        'num_key_value_heads':2,'intermediate_size':12,'vocab_size':11,
        'rms_norm_eps':1e-5,'rope_theta':10000.0,'tie_word_embeddings':True})
    assert complete['selected_next_token']==max(range(11),key=lambda i:ref[i])
    assert complete['range_reads_including_logits']>complete['range_reads_cumulative']
