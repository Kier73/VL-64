"""Single-token 30-layer progression with preserved causal KV and rehoming."""
import copy
import math
import struct
from pathlib import Path

import pytest
from vld import domain_digest
from experiments.model_fabric.execution_dag import compile_decoder_dag
from experiments.model_fabric.full_decoder import StreamingDecoder
from experiments.model_fabric.incremental_decoder import IncrementalDecoder
from experiments.model_fabric.pretrained_orbit import ContentAddressedPages
from test_execution_dag import synthetic_pretrained_ground,qualified_host,scalar_reference


def make(tmp_path,layers=30,dtype='BF16'):
    ground,data,weights=synthetic_pretrained_ground(layers=layers,dtype=dtype)
    plan=compile_decoder_dag(ground)
    store=ContentAddressedPages(tmp_path/'pages')
    a=qualified_host(ground,data,observer='source-A')
    b=qualified_host(ground,data,observer='source-B')
    base=StreamingDecoder(plan,[4,3],lambda i:a,lambda i:{'source-A'},store)
    for _ in range(layers):base.advance_layer()
    return plan,store,base,weights,a,b,ground,data


def scalar(tokens,weights,layers):
    return scalar_reference(tokens,weights,{'num_hidden_layers':layers,
        'hidden_size':8,'num_attention_heads':4,'num_key_value_heads':2,
        'intermediate_size':12,'vocab_size':11,'rms_norm_eps':1e-5,
        'rope_theta':10000.0,'tie_word_embeddings':True})


@pytest.mark.parametrize('dtype',['BF16','F32'])
def test_full_30_layer_generated_token_continuation_without_prefix_recompute(tmp_path,dtype):
    plan,store,base,weights,a,b,_,_=make(tmp_path,dtype=dtype)
    prev=base.checkpoint()
    prev_logits=base.logits()
    next_id=prev_logits['selected_next_token']
    seen=[]
    def on_host(i):
        seen.append(i)
        return b
    runner=IncrementalDecoder(plan,prev,expected_head=prev['checkpoint_digest'],
        host_for=on_host,approved_for=lambda i:{'source-B'},store=store,
        owner='replacement-process',epoch=1)
    assert seen==[]  # Verify retained state BEFORE asking any host for weights.
    advance=runner.append_token(next_id)
    result,logits=runner.logits(return_logits=True)
    ref=scalar([4,3,next_id],weights,30)
    assert advance['position']==2
    assert advance['causal_KV_layers']==30
    assert runner.tokens==[4,3,next_id]
    assert max(abs(a-b) for a,b in zip(logits,ref))<2e-6
    assert result['selected_next_token']==max(range(11),key=lambda i:ref[i])
    assert seen.count(-1)==1
    assert set(seen)==set(range(-1,31))
    assert runner.checkpoint()['previous_checkpoint']==prev['checkpoint_digest']
    assert runner.checkpoint()['weight_identities']==prev['weight_identities']


def test_second_process_and_second_generated_step_reuse_committed_prefix(tmp_path):
    plan,store,base,weights,a,b,_,_=make(tmp_path,layers=3)
    first=IncrementalDecoder(plan,base.checkpoint(),
        expected_head=base.checkpoint()['checkpoint_digest'],
        host_for=lambda i:b,approved_for=lambda i:{'source-B'},
        store=store,owner='worker-B',epoch=1)
    first.append_token(7)
    snap=first.checkpoint()
    second=IncrementalDecoder(plan,snap,expected_head=snap['checkpoint_digest'],
        host_for=lambda i:a,approved_for=lambda i:{'source-A'},
        store=store,owner='worker-A',epoch=2)
    second.append_token(5)
    receipt,logits=second.logits(return_logits=True)
    ref=scalar([4,3,7,5],weights,3)
    assert max(abs(x-y) for x,y in zip(logits,ref))<2e-6
    assert second.checkpoint()['previous_checkpoint']==snap['checkpoint_digest']
    assert receipt['layers_executed']==3


def test_changed_full_layer_weights_are_rejected_before_prefix_commit(tmp_path):
    plan,store,base,_,a,b,ground,data=make(tmp_path,layers=3)
    snap=base.checkpoint()
    def altered(request):
        response=b(request)
        if request['tensor']=='model.layers.1.input_layernorm.weight':
            raw=bytearray(response['payload'])
            raw[:2]=b'\x80\x3f'  # a different finite BF16 number
            import hashlib
            response['payload']=bytes(raw)
            response['sha256']=hashlib.sha256(response['payload']).hexdigest()
        return response
    runner=IncrementalDecoder(plan,snap,expected_head=snap['checkpoint_digest'],
        host_for=lambda i:altered,approved_for=lambda i:{'source-B'},
        store=store,owner='new-process',epoch=3)
    with pytest.raises(ValueError,match='changed pinned'):
        runner.append_token(8)
    assert runner.checkpoint()['previous_checkpoint']==snap['checkpoint_digest']
    assert runner.checkpoint()['tokens']==[4,3]


def test_wrong_expected_head_and_epoch_rejected_before_host_contact(tmp_path):
    plan,store,base,_,a,b,_,_=make(tmp_path,layers=3)
    snap=base.checkpoint()
    with pytest.raises(ValueError,match='checkpoint|frontier'):
        IncrementalDecoder(plan,snap,expected_head='0'*64,
            host_for=lambda i:b,approved_for=lambda i:{'source-B'},
            store=store,owner='new',epoch=1)
    with pytest.raises(ValueError,match='newer'):
        IncrementalDecoder(plan,snap,expected_head=snap['checkpoint_digest'],
            host_for=lambda i:b,approved_for=lambda i:{'source-B'},
            store=store,owner='new',epoch=0)
