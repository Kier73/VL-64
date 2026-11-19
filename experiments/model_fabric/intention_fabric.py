"""Intent-guided operational model Grounds and isolated model instances.

A tokenizer and trained model stay at their host. VL64 chooses *what* task to
navigate, admits its tokenization policy, verifies bounded inference receipts,
and refuses outputs that fail their declared shape. This is not a proof of
semantic quality or a substitute for executing the model's learned arithmetic.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import Callable

from .operational_ground import (
    OperationalModelNavigator, _check_ground, canonical, digest,
    observe_operational_ground,
)

SCHEMA = 'vl64.intent-inference-ground.v1'
OUTPUT_SCHEMA = 'vl64.intent-inference-observation.v1'
CHECKPOINT_SCHEMA = 'vl64.intent-inference-checkpoint.v1'
CHAT_PREFIX = 'VL64_ROLE_MESSAGES_V1:'
KINDS = frozenset(('paragraph', 'python', 'tool_json'))
ROLES = frozenset(('system', 'user', 'assistant', 'tool'))
MAX_MESSAGES = 16
MAX_TASK_BYTES = 8192
MAX_ACTION_BYTES = 8192
_SAFE_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_.-]{0,79}\Z')


def _pairs(items):
    out = {}
    for k, value in items:
        if k in out:
            raise ValueError('Duplicate key in model action')
        out[k] = value
    return out


def _messages(messages):
    if (not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES or
            not all(isinstance(item, dict) and set(item) == {'role', 'content'} and
                    item['role'] in ROLES and isinstance(item['content'], str) and
                    0 < len(item['content'].encode('utf-8')) <= 4096
                    for item in messages) or
            not any(item['role'] == 'user' for item in messages)):
        raise ValueError('Unqualified role-aware dialogue')
    if len(canonical(messages)) > MAX_TASK_BYTES:
        raise ValueError('Dialogue exceeds bounded tokenizer intention')
    return [dict(m) for m in messages]


def compile_intention(model_ground, *, kind, messages, max_new_tokens=64,
                      max_output_bytes=4096, tokenizer_mode='plain_roles', stop_tokens=()):
    """Bind tokenization + inference + output and authority policy to one model.

    The intent does not contain weight bytes or hallucinated model internals.
    The model Ground already binds revision, tokenizer identity and host claim.
    """
    g = _check_ground(dict(model_ground))
    messages = _messages(messages)
    if kind not in KINDS:
        raise ValueError('Unsupported intention type')
    if (tokenizer_mode not in ('plain_roles', 'native_chat') or
            (tokenizer_mode == 'native_chat' and
             g['tokenizer_id'] != 'hf_native_chat_template_v1') or
            type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256 or
            type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 16384 or
            not isinstance(stop_tokens, (list, tuple)) or
            len(stop_tokens) > 16 or
            any(type(t) is not int or not 0 <= t < g['vocab_size'] for t in stop_tokens)):
        raise ValueError('Unadmitted tokenizer/inference budget')
    record = {
        'schema': SCHEMA, 'model_ground_digest': g['ground_digest'],
        'model_artifact_identity': g['model_artifact_identity'],
        'tokenizer_id': g['tokenizer_id'], 'tokenizer_mode': tokenizer_mode,
        'kind': kind, 'messages': messages, 'stop_tokens': list(stop_tokens),
        'max_new_tokens': max_new_tokens, 'max_output_bytes': max_output_bytes,
        'scope': 'host_model_math_and_tokenizer; vl64_intent_navigation_and_receipts',
    }
    record['intent_digest'] = digest(SCHEMA, record)
    return record


def _check_intention(record, ground):
    if (not isinstance(record, dict) or set(record) != {
            'schema','model_ground_digest','model_artifact_identity','tokenizer_id',
            'tokenizer_mode','kind','messages','stop_tokens','max_new_tokens',
            'max_output_bytes','scope','intent_digest'} or
            record.get('schema') != SCHEMA or
            record.get('model_ground_digest') != ground['ground_digest'] or
            record.get('model_artifact_identity') != ground['model_artifact_identity'] or
            record.get('tokenizer_id') != ground['tokenizer_id'] or
            record.get('intent_digest') != digest(SCHEMA,
                {k:v for k,v in record.items() if k != 'intent_digest'})):
        raise ValueError('Intention is not bound to this qualified model')
    expected = compile_intention(ground,kind=record['kind'],
        messages=record['messages'],max_new_tokens=record['max_new_tokens'],
        max_output_bytes=record['max_output_bytes'],tokenizer_mode=record['tokenizer_mode'],
        stop_tokens=record['stop_tokens'])
    if expected != record:
        raise ValueError('Intention fields fail canonical re-admission')


def _instruction(kind):
    return {
        'paragraph': ('Respond with a coherent paragraph answering the latest user request. '
                      'Use complete sentences, not a list of token predictions.'),
        'python': ('Respond with executable Python source code only. No Markdown fences. '
                   'Do not claim the code was run.'),
        'tool_json': ('Return exactly one JSON object: {"tool":"name","arguments":{...}} '
                      'for a caller-approved tool, or {"final":"answer"}. '
                      'Do not execute tools yourself.'),
    }[kind]


def render_intention(record):
    """Protocol-neutral text envelope. Native-chat hosts render their own template.

    JSON escaping ensures user content cannot manufacture protocol role records.
    No fixed special-token IDs are assumed to work across model architectures.
    """
    messages = [{'role':'system','content':_instruction(record['kind'])}] + list(record['messages'])
    if record['tokenizer_mode'] == 'native_chat':
        envelope = {'schema':'vl64.role-messages.v1', 'messages':messages,
                    'add_generation_prompt':True}
        return CHAT_PREFIX + canonical(envelope).decode('utf-8')
    return 'VL64 role messages (JSON):\n' + canonical(messages).decode('utf-8') + '\nAssistant:\n'


def _validate_output(kind, raw):
    cleaned = raw.strip()
    if not cleaned:
        return False, 'empty_model_output', None
    if kind == 'paragraph':
        # Shape check, not an evaluation of accuracy or linguistic competence.
        return (len(cleaned) >= 24 and any(c in cleaned for c in '.!?'),
                'paragraph_shape', cleaned)
    if kind == 'python':
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            return False, 'invalid_python_syntax', None
        return bool(tree.body), 'python_syntax_only_not_executed', cleaned
    try:
        obj = json.loads(cleaned, object_pairs_hook=_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite action')))
    except (ValueError, TypeError):
        return False, 'invalid_action_json', None
    if not isinstance(obj, dict):
        return False, 'action_not_object', None
    if set(obj) == {'final'} and isinstance(obj['final'], str) and obj['final']:
        return True, 'final_decision', obj
    if (set(obj) == {'tool','arguments'} and
            isinstance(obj['tool'], str) and _SAFE_NAME.fullmatch(obj['tool']) and
            isinstance(obj['arguments'], dict) and len(canonical(obj['arguments'])) <= MAX_ACTION_BYTES):
        return True, 'action_proposal_not_executed', obj
    return False, 'invalid_action_shape', None


class IntentionSession:
    """One logical VL model instance: typed intention -> tokenizer -> causal model.

    The host retains weights; this instance owns only tokens, receipts, and intent.
    Host execution claims are not independent proof of trained arithmetic.
    """
    def __init__(self, intent, ground, host: Callable, *, approved_hosts):
        self.ground = _check_ground(dict(ground))
        _check_intention(intent,self.ground)
        self.intent = dict(intent)
        self.nav = OperationalModelNavigator(self.ground,host,approved_hosts=approved_hosts)

    def run(self, *, tile_size=4096):
        tokens = self.nav.encode(render_intention(self.intent))
        if len(tokens) + self.intent['max_new_tokens'] >= self.ground['max_context']:
            raise ValueError('Insufficient admitted causal context for intention')
        self.nav.open_tokens(tokens)
        generated=[]
        for _ in range(self.intent['max_new_tokens']):
            token = self.nav.advance(tile_size=tile_size)
            if token in self.intent['stop_tokens']:
                break
            generated.append(token)
        raw = self.nav.decode(generated)
        if len(raw.encode('utf-8')) > self.intent['max_output_bytes']:
            valid, reason, value = False, 'output_byte_budget', None
        else:
            valid, reason, value = _validate_output(self.intent['kind'],raw)
        observation = {
            'schema':OUTPUT_SCHEMA, 'intent_digest':self.intent['intent_digest'],
            'model_ground_digest':self.ground['ground_digest'],
            'status':'OUTPUT_SHAPE_ADMITTED' if valid else 'OUTPUT_CONTRACT_UNRESOLVED',
            'validation':reason, 'observed_text':raw,
            'parsed_result':value, 'generated_tokens':generated,
            'input_token_count':len(tokens),
            'transcript_head':self.nav.transcript_head,
            'checkpoint_digest':self.nav.checkpoint()['checkpoint_digest'],
            'remote_weight_bytes_received':self.nav.remote_weight_bytes_received,
            'host_calls':self.nav.remote_calls,
            'scope':'host_resident_model_execution_not_semantic_quality_attestation',
        }
        observation['observation_digest'] = digest(OUTPUT_SCHEMA,observation)
        return observation

    def checkpoint(self):
        state = self.nav.checkpoint()
        record={'schema':CHECKPOINT_SCHEMA,
                'intent_digest':self.intent['intent_digest'],
                'model_ground_digest':self.ground['ground_digest'],
                'model_checkpoint':state}
        record['checkpoint_digest']=digest(CHECKPOINT_SCHEMA,record)
        return record


class ModelInstanceFleet:
    """Independently pinned VL-64 instances for different resident model hosts.

    A mapping controls admission and per-model routing. Parallel mode uses
    independent sessions and never shares vocabulary, tokenizer, KV or weights.
    """
    def __init__(self, bindings):
        if not isinstance(bindings, dict) or not bindings or len(bindings)>16:
            raise ValueError('Model fleet requires 1..16 explicit host bindings')
        self.bindings={}
        for key,bind in bindings.items():
            if (not isinstance(key,str) or not _SAFE_NAME.fullmatch(key) or
                    not isinstance(bind,dict) or set(bind)!={'ground','host','approved_hosts'}):
                raise ValueError('Invalid model instance binding')
            ground=_check_ground(dict(bind['ground']))
            if not callable(bind['host']) or not isinstance(bind['approved_hosts'],set) or \
                    ground['observer_host'] not in bind['approved_hosts']:
                raise ValueError('Unqualified model host binding')
            self.bindings[key]={'ground':ground,'host':bind['host'],
                                'approved_hosts':set(bind['approved_hosts'])}

    def execute(self, requests, *, parallel=False):
        if (not isinstance(requests,list) or not 1<=len(requests)<=32 or
                any(not isinstance(q,dict) or set(q)!={'instance','intent'} or
                    q['instance'] not in self.bindings for q in requests)):
            raise ValueError('Unqualified model fleet requests')
        # Duplicate instance names within one batch could race against the same
        # stateful host. Require distinct assignments; callers sequence repeats.
        keys=[q['instance'] for q in requests]
        if len(set(keys))!=len(keys):
            raise ValueError('Duplicate model instance in concurrent batch')
        def work(request):
            bind=self.bindings[request['instance']]
            session=IntentionSession(request['intent'],bind['ground'],bind['host'],
                         approved_hosts=bind['approved_hosts'])
            result=session.run()
            return {'instance':request['instance'],'model':bind['ground']['model'],
                    'result':result,'checkpoint':session.checkpoint()}
        if not parallel or len(requests)==1:
            return [work(q) for q in requests]
        with ThreadPoolExecutor(max_workers=min(8,len(requests))) as pool:
            return list(pool.map(work,requests))


class AuthorizedIntentAgent:
    """Run strict tool decision Grounds with caller-authorized, bounded actions."""
    def __init__(self, fleet, instance, tools, *, max_rounds=3, stop_tokens=()):
        if (not isinstance(fleet,ModelInstanceFleet) or instance not in fleet.bindings or
                not isinstance(tools,dict) or any(not isinstance(k,str) or
                not _SAFE_NAME.fullmatch(k) or not callable(v) for k,v in tools.items()) or
                type(max_rounds) is not int or not 1<=max_rounds<=8):
            raise ValueError('Invalid explicit agent authority')
        self.fleet,self.instance,self.tools,self.max_rounds=fleet,instance,dict(tools),max_rounds
        self.stop_tokens=tuple(stop_tokens)

    def run(self, task, *, max_new_tokens=128):
        if not isinstance(task,str) or not 1<=len(task.encode('utf-8'))<=4096:
            raise ValueError('Invalid authorized agent task')
        messages=[{'role':'user','content':task}]
        events=[]
        for turn in range(self.max_rounds):
            bind=self.fleet.bindings[self.instance]
            intent=compile_intention(bind['ground'],kind='tool_json',messages=messages,
                          max_new_tokens=max_new_tokens,stop_tokens=self.stop_tokens)
            outcome=self.fleet.execute([{'instance':self.instance,'intent':intent}])[0]['result']
            if outcome['status']!='OUTPUT_SHAPE_ADMITTED':
                return {'status':'MODEL_ACTION_UNRESOLVED', 'events':events,
                        'reason':outcome['validation'], 'rounds':turn+1}
            obj=outcome['parsed_result']
            if 'final' in obj:
                return {'status':'BOUNDED_AGENT_COMPLETED','final':obj['final'],
                        'events':events,'rounds':turn+1,
                        'scope':'host_model_decisions_caller_authorized_tools'}
            tool=obj['tool']
            if tool not in self.tools:
                raise ValueError('Model requested an unapproved tool')
            try:result=self.tools[tool](**obj['arguments'])
            except (TypeError,ValueError) as exc:
                raise ValueError('Approved tool rejected model arguments') from exc
            event={'tool':tool,'arguments_digest':digest('vl64.agent-args.v1',obj['arguments']),
                   'result':result}
            if len(canonical(event))>MAX_ACTION_BYTES:
                raise ValueError('Tool result exceeds authority budget')
            events.append(event)
            messages += [{'role':'assistant','content':outcome['observed_text']},
                         {'role':'tool','content':canonical(event).decode('utf-8')}]
            if len(canonical(messages))>MAX_TASK_BYTES or len(messages)>MAX_MESSAGES:
                raise ValueError('Agent dialogue exceeds intention budget')
        return {'status':'AGENT_ROUND_BUDGET','events':events,'rounds':self.max_rounds}
