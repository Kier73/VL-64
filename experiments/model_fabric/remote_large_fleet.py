"""Large pretrained model *remote* intent fleet for the VL-64 control plane.

Neither weights nor logits are computed by this adapter. Two fixed, large model
routes are supported; local CPU work is bounded validation, routing, digests,
result grading, and strict caller-authorized action dispatch. Provider chat APIs
are NOT the stronger VL64 tokenizer/logit-tile operational host protocol.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .operational_ground import canonical, digest
from .intention_fabric import _instruction, _messages, _validate_output

SCHEMA = 'vl64.remote-large-intention-ground.v1'
RECEIPT = 'vl64.remote-large-intention-receipt.v1'
CHECKPOINT = 'vl64.remote-large-intention-checkpoint.v1'
FRAME_LIMIT = 65536
MESSAGE_LIMIT = 8192
MAX_OUTPUT = 8192
# These are *serving routes*, NOT guarantees of a pinned HF Git commit.
ROUTES = {
    'qwen-27b': {
        'architecture': 'Qwen/Qwen3.8-27B',
        'api_model': 'Qwen/Qwen3.8-27B',
        'endpoint': 'https://router.huggingface.co/v1/chat/completions',
        'secret_env': 'HF_TOKEN', 'provider': 'huggingface-inference-providers',
    },
    'deepseek-v41-flash': {
        'architecture': 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
        'api_model': 'deepseek-flash',
        'endpoint': 'https://api.deepseek.com/chat/completions',
        'secret_env': 'DEEPSEEK_API_KEY', 'provider': 'deepseek-official-api',
    },
}
TASKS = {
    'paragraph': {
        'kind': 'paragraph',
        'prompt': ('In one complete paragraph, explain why an observed model-weight '
                   'description is not by itself a trained inference result, and '
                   'explain what an evidence receipt can actually establish.'),
    },
    'python': {
        'kind': 'python',
        'prompt': ('Return Python code only, implementing def safe_divide(a, b): '
                   'return None when b is zero, otherwise return a / b. '
                   'No imports, file access, network, or additional functions.'),
    },
    'tool_json': {
        'kind': 'tool_json',
        'prompt': ('Use only the permitted tool add_numbers to calculate 3 + 4. '
                   'Return exactly {"tool":"add_numbers","arguments":{"a":3,"b":4}}. '
                   'Do not calculate using an unapproved tool.'),
    },
}


def ground(route):
    if route not in ROUTES:
        raise ValueError('Unknown model serving route')
    record = {
        'schema': SCHEMA, 'route': route,
        'model_architecture': ROUTES[route]['architecture'],
        'provider': ROUTES[route]['provider'],
        'api_model_alias': ROUTES[route]['api_model'],
        'endpoint': ROUTES[route]['endpoint'],
        'capability': 'provider_chat_messages_only_no_token_logits_or_kv',
        'weight_identity': 'provider_route_claim_not_pinned_weight_attestation',
        'local_weight_payload_bytes': 0,
        'controller': 'VL64_intention_validation_and_evidence',
    }
    record['ground_digest'] = digest(SCHEMA, record)
    return record


def require_ground(record):
    if not isinstance(record, dict) or set(record) != set(ground(record.get('route'))):
        raise ValueError('Unadmitted remote Ground fields')
    if record != ground(record['route']):
        raise ValueError('Remote Ground identity was altered')
    return record


def intention(record, kind, messages, *, max_tokens=160):
    require_ground(record)
    messages = _messages(messages)
    if kind not in TASKS or type(max_tokens) is not int or not 1 <= max_tokens <= 384:
        raise ValueError('Unadmitted large-model intention')
    obj = {
        'schema': 'vl64.remote-large-intention.v1',
        'ground_digest': record['ground_digest'],
        'kind': kind, 'messages': messages, 'max_tokens': max_tokens,
        'sampling': 'temperature_0_if_provider_accepts',
    }
    obj['intent_digest'] = digest(obj['schema'], obj)
    return obj


def _check_intention(record, spec):
    require_ground(record)
    if not isinstance(spec, dict) or set(spec) != set(intention(record, spec.get('kind'),
                                      spec.get('messages'), max_tokens=spec.get('max_tokens'))):
        raise ValueError('Invalid intent fields')
    if intention(record, spec['kind'], spec['messages'], max_tokens=spec['max_tokens']) != spec:
        raise ValueError('Intent digest or model binding differs')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPSChatTransport:
    """Exact allowlisted HTTPS host. No redirects or credential logging.

    The full incoming JSON frame is capped at 64 KiB. This transport performs
    no CPU inference and never loads a model library or pretrained weights.
    """
    def __init__(self, *, timeout=90):
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
            raise ValueError('Invalid host timeout')
        self.timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect)

    def __call__(self, record, payload):
        route = require_ground(record)['route']
        config = ROUTES[route]
        secret = os.environ.get(config['secret_env'], '')
        if not secret or '\n' in secret or '\r' in secret:
            raise ValueError('Qualified large model requires provider credentials')
        body = canonical(payload)
        if len(body) > MESSAGE_LIMIT:
            raise ValueError('Outbound control-plane frame exceeds budget')
        req = urllib.request.Request(config['endpoint'], data=body, method='POST',
                     headers={'Content-Type':'application/json',
                              'Accept':'application/json',
                              'Authorization':'Bearer '+secret})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                if response.status != 200:
                    raise RuntimeError('Remote model did not complete response')
                raw = response.read(FRAME_LIMIT + 1)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            # Do not echo untrusted response bodies or credentials into reports.
            raise RuntimeError('Qualified remote model endpoint unavailable') from exc
        if not 2 <= len(raw) <= FRAME_LIMIT:
            raise ValueError('Incomplete or oversized remote model response')
        return raw


def _accept_response(raw, route):
    if isinstance(raw, dict):
        raw = canonical(raw)
    if not isinstance(raw, bytes) or not 2 <= len(raw) <= FRAME_LIMIT:
        raise ValueError('Unadmitted remote chat response frame')
    try:
        obj = json.loads(raw.decode('utf-8'),
             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite provider JSON')))
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Malformed remote JSON response') from exc
    if not isinstance(obj, dict) or not isinstance(obj.get('choices'), list) or len(obj['choices']) != 1:
        raise ValueError('No single bounded model response')
    choice = obj['choices'][0]
    if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
        raise ValueError('Missing provider assistant message')
    content = choice['message'].get('content')
    if not isinstance(content, str) or not content.strip() or len(content.encode('utf-8')) > MAX_OUTPUT:
        raise ValueError('Empty or oversized provider output')
    served = obj.get('model')
    if not isinstance(served, str) or not 1 <= len(served) <= 200:
        raise ValueError('Provider did not identify its serving model')
    allowed=(('Qwen/Qwen3.8-27B',) if route=='qwen-27b' else
             ('deepseek-flash','DeepSeek-V4.1-Flash-1017',
              'deepseek-ai/DeepSeek-V4.1-Flash-1017'))
    if not any(served==alias or served.startswith(alias+':') for alias in allowed):
        raise ValueError('Provider reported a different model identity')
    usage = obj.get('usage') or {}
    if not isinstance(usage, dict):
        raise ValueError('Invalid provider usage evidence')
    counts = {}
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = usage.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Invalid token usage count')
        counts[field] = value
    return content, served, counts, len(raw), digest('vl64.chat-provider-raw.v1', raw)


def _python_task_quality(text):
    """Fail-closed AST requirements; never execute model-generated Python."""
    try: tree = ast.parse(text)
    except SyntaxError: return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    func = tree.body[0]
    if func.name != 'safe_divide' or [a.arg for a in func.args.args] != ['a','b'] or func.decorator_list:
        return False
    safe = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.Compare,
            ast.Eq, ast.Constant, ast.Name, ast.Load, ast.Return, ast.BinOp, ast.Div, ast.Pass)
    if any(not isinstance(node, safe) for node in ast.walk(tree)):
        return False
    # AST shape is necessary but not sufficient; deliberately no eval/exec.
    return any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div)
               for x in ast.walk(func)) and any(isinstance(x, ast.Constant) and x.value is None
               for x in ast.walk(func))


def _quality(kind, text):
    if kind == 'paragraph':
        lower = text.lower()
        return (len(text.split()) >= 28 and 'weight' in lower and
                'infer' in lower and 'evidence' in lower)
    if kind == 'python':
        return _python_task_quality(text)
    if kind == 'tool_json':
        return False  # Only approved tool dispatch can establish agent completion.
    raise ValueError('Unknown benchmark quality assessment')


class RemoteLargeSession:
    """A VL64 interpreter session with provider-side model arithmetic.

    Durable state consists of intent, messages, verified observations and
    digests. A chat API does not export host KV and cannot promise exact replay.
    """
    def __init__(self, record, transport: Callable):
        self.ground = dict(require_ground(record))
        if not callable(transport):
            raise ValueError('Transport must be a callable endpoint adapter')
        self.transport = transport
        self.history = []
        self.head = '0'*64
        self.epoch = 0
        self.owner = 'initial'

    def run(self, spec):
        _check_intention(self.ground, spec)
        kind = spec['kind']
        messages = [{'role':'system','content':_instruction(kind)}] + list(spec['messages'])
        if len(canonical(messages)) > MESSAGE_LIMIT:
            raise ValueError('Model intent exceeds allowed host-request frame')
        payload = {'model': ROUTES[self.ground['route']]['api_model'],
                   'messages': messages, 'max_tokens':spec['max_tokens'],
                   'temperature': 0, 'stream':False}
        control_start=time.perf_counter()
        cpu_start=time.process_time()
        raw_request=canonical(payload)
        if len(raw_request)>MESSAGE_LIMIT:
            raise ValueError('Outbound intention frame exceeds 8 KiB')
        start_network=time.perf_counter()
        raw=self.transport(self.ground,payload)
        host_wall=time.perf_counter()-start_network
        text, served, counts, incoming_bytes, provider_digest = _accept_response(raw,self.ground['route'])
        shape, validation, parsed = _validate_output(kind,text)
        quality=_quality(kind,text) if shape else False
        evidence = {'schema': RECEIPT, 'model_ground_digest':self.ground['ground_digest'],
                    'intent_digest':spec['intent_digest'], 'route':self.ground['route'],
                    'provider_model_reported':served, 'provider_response_digest':provider_digest,
                    'status':'OUTPUT_SHAPE_ADMITTED' if shape else 'OUTPUT_CONTRACT_UNRESOLVED',
                    'validation':validation, 'quality_heuristic_pass':quality,
                    'text':text, 'parsed':parsed,
                    'usage':counts, 'request_bytes':len(raw_request),
                    'response_bytes':incoming_bytes, 'remote_weight_bytes_received':0,
                    'provider_wall_seconds':round(host_wall,6),
                    'vl_cpu_seconds':round(time.process_time()-cpu_start,6),
                    'total_wall_seconds':round(time.perf_counter()-control_start,6),
                    'provider_completion_tokens_per_wall_second':(
                        round(counts['completion_tokens']/host_wall,6)
                        if counts['completion_tokens'] is not None and host_wall>0 else None),
                    'execution_scope':'provider_chat_delegation_no_observed_token_logits',
                    'weight_scope':'provider_model_route_claim_not_exact_weight_attestation'}
        evidence['previous_head']=self.head
        evidence['receipt_digest']=digest(RECEIPT,evidence)
        # Atomic publication only after strict response admission.
        self.history.append(evidence)
        self.head=digest('vl64.remote-large-transcript.v1',
                       {'previous':self.head, 'receipt_digest':evidence['receipt_digest']})
        return evidence

    def checkpoint(self):
        state={'schema':CHECKPOINT,'ground_digest':self.ground['ground_digest'],
               'owner':self.owner,'epoch':self.epoch, 'transcript_head':self.head,
               'history':[dict(r) for r in self.history]}
        state['checkpoint_digest']=digest(CHECKPOINT,state)
        return state

    @classmethod
    def restore(cls,record,transport,checkpoint,*,expected_digest,owner,epoch):
        require_ground(record)
        if (not isinstance(checkpoint,dict) or set(checkpoint) != {
                'schema','ground_digest','owner','epoch','transcript_head',
                'history','checkpoint_digest'} or checkpoint.get('schema')!=CHECKPOINT or
                checkpoint.get('ground_digest')!=record['ground_digest'] or
                checkpoint.get('checkpoint_digest')!=expected_digest or
                digest(CHECKPOINT,{k:v for k,v in checkpoint.items() if k!='checkpoint_digest'})!=expected_digest or
                type(epoch) is not int or type(checkpoint.get('epoch')) is not int or
                epoch<=checkpoint['epoch'] or not isinstance(owner,str) or not owner or
                not isinstance(checkpoint.get('history'),list) or len(checkpoint['history'])>64):
            raise ValueError('Untrusted or stale remote intent checkpoint')
        h='0'*64
        for event in checkpoint['history']:
            if (not isinstance(event,dict) or event.get('previous_head')!=h or
                event.get('receipt_digest')!=digest(RECEIPT,
                   {k:v for k,v in event.items() if k!='receipt_digest'}) or
                event.get('model_ground_digest')!=record['ground_digest']):
                raise ValueError('Unqualified event in retained intent transcript')
            h=digest('vl64.remote-large-transcript.v1',
                     {'previous':h,'receipt_digest':event['receipt_digest']})
        if h!=checkpoint['transcript_head']:
            raise ValueError('Remote transcript head differs from external pin')
        new=cls(record,transport)
        new.history=json.loads(canonical(checkpoint['history']))
        new.head=h
        new.epoch=epoch;new.owner=owner
        return new


def agent_action(session, *, max_tokens=128):
    """Exactly one proposed add_numbers action and one provider finalization.

    No filesystem, shell or network authority is delegated to the model.
    """
    record=session.ground
    spec=intention(record,'tool_json',[{'role':'user','content':TASKS['tool_json']['prompt']}],max_tokens=max_tokens)
    first=session.run(spec)
    if first['status']!='OUTPUT_SHAPE_ADMITTED' or not isinstance(first['parsed'],dict):
        return {'status':'ACTION_UNRESOLVED', 'reason':'invalid_action_proposal','receipts':[first]}
    action=first['parsed']
    if action != {'tool':'add_numbers','arguments':{'a':3,'b':4}}:
        return {'status':'ACTION_UNRESOLVED','reason':'unapproved_or_incorrect_action','receipts':[first]}
    # In-process pure authorized tool: no dynamic eval or execution.
    result=action['arguments']['a']+action['arguments']['b']
    follow=intention(record,'tool_json',[
        {'role':'user','content':TASKS['tool_json']['prompt']},
        {'role':'assistant','content':first['text']},
        {'role':'user','content':f'Authorized add_numbers returned {result}. Respond exactly {{"final":"{result}"}}.'},
    ],max_tokens=max_tokens)
    second=session.run(follow)
    passed=second['status']=='OUTPUT_SHAPE_ADMITTED' and second['parsed']=={'final':str(result)}
    return {'status':'AUTHORIZED_ACTION_COMPLETED' if passed else 'ACTION_UNRESOLVED',
            'tool':'add_numbers','tool_result':result,
            'receipts':[first,second],
            'authority':'explicit_add_numbers_only;no_unapproved_side_effects'}


def fixture_transport(route):
    """Deterministic *protocol-only* fixture; not a pretrained model."""
    require_ground(ground(route))
    def respond(record,payload):
        if record['route']!=route or payload['model']!=ROUTES[route]['api_model']:
            raise ValueError('Model transport crossed fleet Ground identity')
        instruction=payload['messages'][0]['content']
        if 'coherent paragraph' in instruction:
            out=('A weight description records parameters, but inference requires executing '
                 'the trained mathematical operations across input tokens. An evidence '
                 'receipt can establish the observed outputs and their source claims, '
                 'not independently prove an unseen provider performed the correct work.')
        elif 'executable Python source' in instruction:
            out='def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n'
        elif len(payload['messages'])>2:
            out='{"final":"7"}'
        else:
            out='{"tool":"add_numbers","arguments":{"a":3,"b":4}}'
        return canonical({'model':payload['model'], 'choices':[{'message':{'role':'assistant','content':out}}],
                          'usage':{'prompt_tokens':17,'completion_tokens':11,'total_tokens':28}})
    return respond


def evaluate_fleet(*, transport_factory=None, live=False, max_tokens=160, routes=None):
    """Run *both full-scale named routes* in isolated logical VL sessions.

    In tests, transport_factory supplies a protocol fixture. For live model
    execution, the only permitted transport is the authenticated HTTPS route.
    """
    if routes is None:
        routes=tuple(ROUTES)
    if (not isinstance(routes,(list,tuple)) or not routes or len(routes)>2 or
            any(r not in ROUTES for r in routes) or len(set(routes))!=len(routes)):
        raise ValueError('Only exact, distinct large-model routes permitted')
    if live and transport_factory is not None:
        raise ValueError('Live mode requires authenticated HTTPS provider transport')
    if transport_factory is None:
        if not live:
            raise ValueError('Specify fixture transport or explicit live provider mode')
        shared=HTTPSChatTransport()
        transport_factory=lambda _route:shared
    before=time.perf_counter()
    def execute(route):
        session=RemoteLargeSession(ground(route),transport_factory(route))
        outputs={}
        for kind in ('paragraph','python'):
            spec=intention(session.ground,kind,[{'role':'user','content':TASKS[kind]['prompt']}],
                           max_tokens=max_tokens)
            outputs[kind]=session.run(spec)
        outputs['agent']=agent_action(session,max_tokens=max_tokens)
        return {'route':route,'architecture':session.ground['model_architecture'],
                'ground_digest':session.ground['ground_digest'],
                'tasks':outputs, 'checkpoint_digest':session.checkpoint()['checkpoint_digest']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations=list(pool.map(execute,routes))
    return {'schema':'vl64.large-instruction-fleet-benchmark.v1',
            'status':'REAL_PROVIDER_CHAT_OBSERVATIONS' if live else 'CONTROL_PLANE_FIXTURE_ONLY',
            'instances':observations,'fleet_wall_seconds':round(time.perf_counter()-before,6),
            'weight_bytes_held_by_vl':0,
            'claim':'remote_chat_intent_not_direct_VL64_token_logit_control'}
