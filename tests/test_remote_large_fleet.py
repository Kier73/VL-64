"""Two large model *control-plane* contracts, without loading any model."""
import json
import os
import subprocess
import sys
import copy

import pytest
from experiments.model_fabric.remote_large_fleet import (
    FRAME_LIMIT, CHECKPOINT, ROUTES, HTTPSChatTransport, RemoteLargeSession,
    _accept_response, _python_task_quality, agent_action, evaluate_fleet,
    fixture_transport, ground, intention, TASKS,
)
from experiments.model_fabric.operational_ground import canonical


def test_only_large_model_serving_routes_are_admitted():
    assert set(ROUTES)=={'qwen-27b','deepseek-v41-flash'}
    assert ROUTES['qwen-27b']['architecture']=='Qwen/Qwen3.8-27B'
    assert ROUTES['deepseek-v41-flash']['architecture']=='deepseek-ai/DeepSeek-V4.1-Flash-1017'
    for invalid in ('smollm2', 'qwen-0.5b', 'llama-7b', ''):
        with pytest.raises(ValueError):ground(invalid)


def test_no_weight_data_or_token_level_inference_is_claimed():
    r=evaluate_fleet(transport_factory=fixture_transport)
    assert r['status']=='CONTROL_PLANE_FIXTURE_ONLY'
    assert r['weight_bytes_held_by_vl']==0
    assert len(r['instances'])==2
    for item in r['instances']:
        g=ground(item['route'])
        assert g['local_weight_payload_bytes']==0
        assert g['capability']=='provider_chat_messages_only_no_token_logits_or_kv'
        assert item['tasks']['paragraph']['status']=='OUTPUT_SHAPE_ADMITTED'
        assert item['tasks']['paragraph']['quality_heuristic_pass'] is True
        assert item['tasks']['python']['quality_heuristic_pass'] is True
        assert item['tasks']['agent']['status']=='AUTHORIZED_ACTION_COMPLETED'
        assert item['tasks']['agent']['tool_result']==7
        for key in ('paragraph','python'):
            evidence=item['tasks'][key]
            assert evidence['remote_weight_bytes_received']==0
            assert evidence['usage']['completion_tokens']==11
            assert evidence['execution_scope'].startswith('provider_chat')
            assert evidence['provider_wall_seconds']>=0
            assert evidence['vl_cpu_seconds']>=0


def test_recovery_from_externally_pinned_head_does_not_replay_model_tokens():
    route='qwen-27b'
    s=RemoteLargeSession(ground(route), fixture_transport(route))
    spec=intention(s.ground,'paragraph',[{'role':'user','content':TASKS['paragraph']['prompt']}])
    one=s.run(spec)
    checkpoint=s.checkpoint()
    assert checkpoint['checkpoint_digest'] and one['receipt_digest']
    new=RemoteLargeSession.restore(s.ground,fixture_transport(route),checkpoint,
            expected_digest=checkpoint['checkpoint_digest'],owner='new-controller',epoch=1)
    assert new.head==s.head
    assert len(new.history)==1
    assert new.run(spec)['previous_head']==s.head
    assert len(s.history)==1


def test_tampered_checkpoint_and_stale_epoch_refuse():
    g=ground('deepseek-v41-flash')
    s=RemoteLargeSession(g,fixture_transport('deepseek-v41-flash'))
    spec=intention(g,'python',[{'role':'user','content':TASKS['python']['prompt']}])
    s.run(spec)
    cp=s.checkpoint()
    tampered=copy.deepcopy(cp);tampered['history'][0]['text']='malicious'
    for bad,ep in ((tampered,2),(cp,0)):
        with pytest.raises(ValueError):
            RemoteLargeSession.restore(g,fixture_transport(g['route']),bad,
                    expected_digest=cp['checkpoint_digest'],owner='new',epoch=ep)


def test_malformed_ground_and_cross_model_intention_refuse():
    g=ground('qwen-27b');bad=copy.deepcopy(g)
    bad['model_architecture']='small-model'
    with pytest.raises(ValueError):RemoteLargeSession(bad,fixture_transport('qwen-27b'))
    intent=intention(g,'python',[{'role':'user','content':'code'}])
    other=RemoteLargeSession(ground('deepseek-v41-flash'),fixture_transport('deepseek-v41-flash'))
    with pytest.raises(ValueError):other.run(intent)


def test_refuses_oversized_nonfinite_and_missing_provider_identity():
    raw={'choices':[{'message':{'content':'a'}}], 'model':'Qwen/Qwen3.8-27B'}
    assert _accept_response(raw,'qwen-27b')[0]=='a'
    raw['model']='another/smaller-model'
    with pytest.raises(ValueError):_accept_response(raw,'qwen-27b')
    raw['model']='Qwen/Qwen3.8-27B'
    raw['choices'][0]['message']['content']='x' * FRAME_LIMIT
    with pytest.raises(ValueError):_accept_response(raw,'qwen-27b')
    assert not _python_task_quality('import os\nprint(os.getenv("SECRET"))')
    assert not _python_task_quality('def safe_divide(a,b):\n    return eval("a/b")')


def test_real_http_transport_requires_provider_key(monkeypatch):
    monkeypatch.delenv('HF_TOKEN',raising=False)
    t=HTTPSChatTransport()
    with pytest.raises(ValueError,match='credentials'):
        t(ground('qwen-27b'),{'model':'Qwen/Qwen3.8-27B','messages':[]})


def test_action_approval_never_runs_unlisted_tool():
    route='deepseek-v41-flash'
    def malicious(record,payload):
        return canonical({'model':payload['model'],'choices':[{'message':{
            'content':'{"tool":"shell","arguments":{"command":"rm -rf /"}}'}}]})
    session=RemoteLargeSession(ground(route),malicious)
    r=agent_action(session)
    assert r['status']=='ACTION_UNRESOLVED'
    assert r['reason']=='unapproved_or_incorrect_action'


def test_two_separate_interpreter_processes(tmp_path):
    root=tmp_path/'one.json';other=tmp_path/'two.json'
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([os.getcwd(),env.get('PYTHONPATH','')])
    args=[sys.executable,'-m','experiments.model_fabric.remote_large_fleet_lab',
          '--fixture','--max-tokens','64']
    routes=['qwen-27b','deepseek-v41-flash']
    processes=[subprocess.Popen(args+['--instance',route,'--out',str(file)],env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    for file,route in zip((root,other),routes)]
    digests=[]
    for proc,file,route in zip(processes,(root,other),routes):
        output,error=proc.communicate(timeout=20)
        assert proc.returncode==0,error
        report=json.loads(file.read_text())
        assert report['status']=='CONTROL_PLANE_FIXTURE_ONLY'
        assert len(report['instances'])==1
        assert report['instances'][0]['route']==route
        digests.append(report['instances'][0]['checkpoint_digest'])
    assert len(set(digests))==2


def test_remote_validation_refuses_wrong_served_model_and_never_retries():
    route='deepseek-v41-flash'
    called=[]
    def wrong(record,payload):
        called.append(1)
        return canonical({'model':'small substitute',
                          'choices':[{'message':{'content':'Some words.'}}]})
    session=RemoteLargeSession(ground(route),wrong)
    spec=intention(session.ground,'paragraph',[{'role':'user','content':'Explain a Ground.'}])
    with pytest.raises(ValueError,match='different model'):
        session.run(spec)
    assert len(called)==1
    assert len(session.history)==0 and session.head=='0'*64


def test_invalid_agent_json_is_not_executable_authority():
    route='qwen-27b'
    def wrong(record,payload):
        return canonical({'model':'Qwen/Qwen3.8-27B',
            'choices':[{'message':{'content':'{"tool":"add_numbers","arguments":{"a":3,"b":5}}'}}]})
    result=agent_action(RemoteLargeSession(ground(route),wrong))
    assert result['status']=='ACTION_UNRESOLVED'
    assert result['reason']=='unapproved_or_incorrect_action'


def test_protocol_only_does_not_allow_uncredentialed_live_mode():
    with pytest.raises(ValueError,match='requires authenticated HTTPS'):
        evaluate_fleet(live=True,transport_factory=fixture_transport)
