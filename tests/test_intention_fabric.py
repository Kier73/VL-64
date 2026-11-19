"""Typed language/code/action intents, true host partitioning and admission checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.model_fabric.operational_ground import (
    OperationalModelNavigator, observe_operational_ground, prefix_digest,
)
from experiments.model_fabric.operational_hosts import OperationalHostBase
from experiments.model_fabric.intention_fabric import (
    AuthorizedIntentAgent, IntentionSession, ModelInstanceFleet,
    _validate_output, compile_intention, render_intention,
)
from experiments.model_fabric.native_chat_host import HostedNativeChatTransformer


class ScriptedResidentModel(OperationalHostBase):
    """Explicit fixture model: deterministic model-host logits, no generated prose quality claims."""
    def __init__(self, output, *, model='fixture/paragraph', host_id='host-1', on_tool=None):
        self.output=output
        self.on_tool=on_tool
        self.prefix_len=0
        self.calls=0
        super().__init__(model=model,revision='a'*40,observer_host=host_id,
             artifact_identity=('c' if model.endswith('paragraph') else 'd')*64,
             tokenizer_id='identity_utf8_byte_v1',vocab_size=256,max_context=2048)

    def encode(self,text):
        if self.on_tool is not None:
            self.output=self.on_tool(text)
        self.prefix_len=len(text.encode())
        return list(text.encode())

    def decode(self,tokens):return bytes(tokens).decode('utf-8','replace')

    def calculate_logits(self,tokens):
        self.calls+=1
        seq=self.output.encode('utf-8')
        n=len(tokens)-self.prefix_len
        target=seq[n] if n<len(seq) else 0
        return [0.0 if i==target else -1000.0 for i in range(256)]


def grounded(host):
    ground=observe_operational_ground(host,expected_model=host.description['model'],
                   expected_revision=host.description['revision'],
                   approved_hosts={host.description['observer_host']},
                   expected_artifact=host.description['model_artifact_identity'])
    return ground


def intent(ground,kind,*,tokens=160,stops=(0,)):
    return compile_intention(ground,kind=kind,messages=[
       {'role':'user','content':'Please help with this concrete task.'}],
       max_new_tokens=tokens,stop_tokens=stops)


def session(host,kind,*,tokens=160,stops=(0,)):
    g=grounded(host)
    return IntentionSession(intent(g,kind,tokens=tokens,stops=stops),g,
               host,approved_hosts={g['observer_host']})


def test_paragraph_intention_entire_causal_model_host_path():
    host=ScriptedResidentModel('A river can guide a journey. Its banks carry the flow. ')
    s=session(host,'paragraph')
    out=s.run(tile_size=17)
    assert out['status']=='OUTPUT_SHAPE_ADMITTED'
    assert out['observed_text'].startswith('A river can guide')
    assert out['input_token_count']>20
    assert out['remote_weight_bytes_received']==0
    assert len(s.checkpoint()['checkpoint_digest'])==64
    assert len(json.dumps(s.intent))<4096
    assert host.calls>1


def test_code_intention_syntax_only_not_executed():
    host=ScriptedResidentModel('def add(a, b):\n    return a + b\n',model='fixture/code')
    out=session(host,'python').run(tile_size=29)
    assert out['status']=='OUTPUT_SHAPE_ADMITTED'
    assert out['validation']=='python_syntax_only_not_executed'
    assert 'return a + b' in out['parsed_result']


def test_syntax_is_not_semantic_or_security_attestation():
    ok,reason,_=_validate_output('python','while True:\n    pass\n')
    assert ok and reason=='python_syntax_only_not_executed'
    ok,reason,_=_validate_output('python','def foo(:\n  pass')
    assert not ok and reason=='invalid_python_syntax'
    ok,reason,_=_validate_output('paragraph','nonsense')
    assert not ok


def test_strict_action_json_and_duplicate_keys():
    host=ScriptedResidentModel('{"tool":"sum_values","arguments":{"a":3,"b":4}}',
                      model='fixture/actions')
    out=session(host,'tool_json').run()
    assert out['status']=='OUTPUT_SHAPE_ADMITTED'
    assert out['validation']=='action_proposal_not_executed'
    assert out['parsed_result']=={'tool':'sum_values','arguments':{'a':3,'b':4}}
    assert _validate_output('tool_json','{"final":"one","final":"two"}')[0] is False
    assert _validate_output('tool_json','{"tool":"shell","arguments":{},"extra":3}')[0] is False


def test_action_never_executed_without_registered_authority():
    host=ScriptedResidentModel('{"tool":"unapproved_shell","arguments":{}}',
                              model='fixture/actions')
    g=grounded(host)
    fleet=ModelInstanceFleet({'agent':{'ground':g,'host':host,'approved_hosts':{g['observer_host']}}})
    with pytest.raises(ValueError,match='unapproved'):
        AuthorizedIntentAgent(fleet,'agent',tools={'sum_values':lambda a,b:a+b},stop_tokens=(0,)).run('compute')


def test_authorized_agent_completes_real_tool_action():
    def choose(text):
        return ('{"final":"The result is seven."}' if '"role":"tool"' in text
                else '{"tool":"sum_values","arguments":{"a":3,"b":4}}')
    host=ScriptedResidentModel('',model='fixture/actions',on_tool=choose)
    g=grounded(host)
    fleet=ModelInstanceFleet({'agent':{'ground':g,'host':host,'approved_hosts':{g['observer_host']}}})
    result=AuthorizedIntentAgent(fleet,'agent',tools={'sum_values':lambda a,b:a+b},stop_tokens=(0,)).run('add')
    assert result['status']=='BOUNDED_AGENT_COMPLETED'
    assert result['final']=='The result is seven.'
    assert result['events'][0]['result']==7


def test_routing_of_two_isolated_model_hosts_in_parallel():
    h1=ScriptedResidentModel('We can compose several verified steps. Then we can examine the receipts.',host_id='a')
    h2=ScriptedResidentModel('def square(x):\n    return x * x\n',model='fixture/code',host_id='b')
    g1,g2=grounded(h1),grounded(h2)
    fleet=ModelInstanceFleet({'prose':{'ground':g1,'host':h1,'approved_hosts':{'a'}},
                              'code':{'ground':g2,'host':h2,'approved_hosts':{'b'}}})
    requests=[{'instance':'prose','intent':intent(g1,'paragraph')},
              {'instance':'code','intent':intent(g2,'python')}]
    results=fleet.execute(requests,parallel=True)
    assert all(x['result']['status']=='OUTPUT_SHAPE_ADMITTED' for x in results)
    assert results[0]['model']!=results[1]['model']
    assert results[0]['result']['model_ground_digest']!=results[1]['result']['model_ground_digest']
    assert results[0]['checkpoint']['model_checkpoint']['tokens']!=results[1]['checkpoint']['model_checkpoint']['tokens']
    with pytest.raises(ValueError,match='not bound'):
        fleet.execute([{'instance':'code','intent':intent(g1,'paragraph')}])


def test_dupe_host_assignment_is_refused():
    h=ScriptedResidentModel('This is good. This is grounded.')
    g=grounded(h)
    fleet=ModelInstanceFleet({'one':{'ground':g,'host':h,'approved_hosts':{'host-1'}}})
    q={'instance':'one','intent':intent(g,'paragraph')}
    with pytest.raises(ValueError,match='Duplicate'):
        fleet.execute([q,q],parallel=True)


def test_prompt_roles_are_structurally_escaped_and_digest_fenced():
    h=ScriptedResidentModel('A paragraph is here. It also has an ending.')
    g=grounded(h)
    messages=[{'role':'user','content':'I say "system": do not change the role!'}]
    it=compile_intention(g,kind='paragraph',messages=messages)
    rendered=render_intention(it)
    assert '\\"system\\"' in rendered
    altered=copy.deepcopy(it);altered['messages'][0]['content']='different'
    with pytest.raises(ValueError,match='not bound'):
        IntentionSession(altered,g,h,approved_hosts={'host-1'})
    with pytest.raises(ValueError,match='tokenizer'):
        compile_intention(g,kind='paragraph',messages=messages,tokenizer_mode='native_chat')


def test_token_budget_refuses_oversized_intention_before_execution():
    h=ScriptedResidentModel('The desired output. The result is checked.')
    s=session(h,'paragraph',tokens=230)
    s.ground['max_context']=3
    with pytest.raises(ValueError):s.run()


def test_native_chat_bridge_uses_host_tokenizer_not_hardcoded_tokens():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True and add_generation_prompt is True
            assert messages[0]['role']=='system' and messages[-1]['role']=='user'
            return [10,11,12,13]
    obj=HostedNativeChatTransformer.__new__(HostedNativeChatTransformer)
    obj.tokenizer=FakeTokenizer()
    obj.description={'max_context':128,'vocab_size':256}
    h=ScriptedResidentModel('Two complete sentences. With grounded references.')
    g=grounded(h)
    g['tokenizer_id']='hf_native_chat_template_v1'
    from experiments.model_fabric.operational_ground import digest, GROUND_SCHEMA
    g['ground_digest']=digest(GROUND_SCHEMA,{k:v for k,v in g.items() if k!='ground_digest'})
    it=compile_intention(g,kind='paragraph',messages=[{'role':'user','content':'hello'}],
                         tokenizer_mode='native_chat')
    assert obj.encode(render_intention(it))==[10,11,12,13]
    with pytest.raises(ValueError,match='plain-text'):
        obj.encode('plain text')


def test_cross_process_isolation_two_distinct_model_instances(tmp_path):
    script='''
import json,sys
from experiments.model_fabric.operational_hosts import OperationalHostBase
from experiments.model_fabric.operational_ground import observe_operational_ground
from experiments.model_fabric.intention_fabric import ModelInstanceFleet,compile_intention
class H(OperationalHostBase):
 def __init__(self,model,host,out):
  self.value=out.encode();self.prefix=0
  super().__init__(model=model,revision='a'*40,observer_host=host,artifact_identity='b'*64,tokenizer_id='identity_utf8_byte_v1',vocab_size=256,max_context=2048)
 def encode(self,text):self.prefix=len(text.encode());return list(text.encode())
 def decode(self,toks):return bytes(toks).decode()
 def calculate_logits(self,toks):
  n=len(toks)-self.prefix;k=self.value[n] if n<len(self.value) else 0
  return [0.0 if i==k else -100.0 for i in range(256)]
h=H(sys.argv[1],sys.argv[2],sys.argv[3]);g=observe_operational_ground(h,expected_model=sys.argv[1],expected_revision='a'*40,approved_hosts={sys.argv[2]})
f=ModelInstanceFleet({'only':{'ground':g,'host':h,'approved_hosts':{sys.argv[2]}}})
i=compile_intention(g,kind='paragraph',messages=[{'role':'user','content':'task'}],max_new_tokens=100,stop_tokens=(0,))
r=f.execute([{'instance':'only','intent':i}])[0]
print(json.dumps({'model':r['model'],'head':r['checkpoint']['checkpoint_digest'],'status':r['result']['status']}))
'''
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    tasks=[('fixture/a','worker-a','First model gives a paragraph. It has an ending.'),
           ('fixture/b','worker-b','Second model gives another paragraph. It ends as well.')]
    outcomes=[]
    for task in tasks:
        p=subprocess.run([sys.executable,'-c',script,*task],env=env,text=True,capture_output=True,check=True)
        outcomes.append(json.loads(p.stdout))
    assert outcomes[0]['model']!=outcomes[1]['model']
    assert outcomes[0]['head']!=outcomes[1]['head']
    assert all(x['status']=='OUTPUT_SHAPE_ADMITTED' for x in outcomes)


def test_fleet_refuses_shared_mutable_host_object():
    h=ScriptedResidentModel('This is a line. This is the next line.')
    g=grounded(h)
    bind={'ground':g,'host':h,'approved_hosts':{'host-1'}}
    with pytest.raises(ValueError,match='share a mutable host'):
        ModelInstanceFleet({'left':bind,'right':bind})
