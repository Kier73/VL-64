"""Operational model Ground: navigate host-resident model functions without copying weights.

A Ground is a capability description, never a mathematical reconstruction of
arbitrary trained parameters. Host-side numerical execution is still required.
The navigator observes complete, bounded F32 logit tiles and independently
selects the next token, but cannot authenticate an untrusted remote processor
or prove the host computed those logits from its advertised checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Callable

GROUND_SCHEMA = 'vl64.operational-model-ground.v1'
CALL_SCHEMA = 'vl64.operational-model-call.v1'
RESULT_SCHEMA = 'vl64.operational-model-result.v1'
CHECKPOINT_SCHEMA = 'vl64.operational-model-checkpoint.v1'
MAX_FRAME_BYTES = 65536
MAX_LOGITS_PER_TILE = 4096
MAX_TOKENS = 4096
MAX_TEXT_BYTES = 16384
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}\Z')
_OPERATIONS = ('encode', 'decode', 'open', 'logits_tile', 'advance')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(domain, value):
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(domain.encode('utf-8') + b'\x00' + raw).hexdigest()


def prefix_digest(tokens):
    return digest('vl64.operational-token-prefix.v1', tokens)


def _check_tokens(tokens, vocab, *, allow_empty=False):
    if (not isinstance(tokens, list) or len(tokens) > MAX_TOKENS or
            (not allow_empty and not tokens) or
            any(type(t) is not int or not 0 <= t < vocab for t in tokens)):
        raise ValueError('Unadmitted model token sequence')


def _check_ground(g):
    if not isinstance(g, dict) or set(g) != {
        'schema','model','revision','observer_host','tokenizer_id',
        'model_artifact_identity','vocab_size','max_context','operations',
        'trust_scope','ground_digest'}:
        raise ValueError('Malformed operational model Ground')
    if (g['schema'] != GROUND_SCHEMA or g['trust_scope'] !=
            'host_capability_observation_not_remote_weight_or_execution_attestation' or
            g['operations'] != list(_OPERATIONS) or
            any(not isinstance(g[k], str) or not _ID.fullmatch(g[k]) for k in
                ('model', 'revision', 'observer_host', 'tokenizer_id')) or
            type(g['vocab_size']) is not int or not 2 <= g['vocab_size'] <= 10_000_000 or
            type(g['max_context']) is not int or not 1 <= g['max_context'] <= MAX_TOKENS or
            not isinstance(g['model_artifact_identity'], str) or
            not _HEX.fullmatch(g['model_artifact_identity']) or
            g['ground_digest'] != digest(GROUND_SCHEMA,
                {k:v for k,v in g.items() if k != 'ground_digest'})):
        raise ValueError('Unqualified operational model Ground')
    return g


def observe_operational_ground(host: Callable, *, expected_model: str,
                               expected_revision: str, approved_hosts: set,
                               expected_artifact: str | None = None):
    """Admit a *claimed* host capability under external model/host pins.

    Host identity here is a string checked against an external allowlist. It is
    not a signature, TLS channel binding, or a proof of the checkpoint bytes.
    """
    if not callable(host) or not approved_hosts:
        raise ValueError('Missing observer-host or approved identities')
    description = host({'schema': CALL_SCHEMA, 'operation': 'describe'})
    if not isinstance(description, dict):
        raise ValueError('Missing host description')
    g = {k:description.get(k) for k in (
        'schema','model','revision','observer_host','tokenizer_id',
        'model_artifact_identity','vocab_size','max_context','operations','trust_scope')}
    if (g['model'] != expected_model or g['revision'] != expected_revision or
            g['observer_host'] not in approved_hosts or
            (expected_artifact is not None and
             g['model_artifact_identity'] != expected_artifact)):
        raise ValueError('Host model, revision, or artifact does not match external pin')
    g['ground_digest'] = digest(GROUND_SCHEMA, g)
    return _check_ground(g)


class OperationalModelNavigator:
    """Stateless-weight client, single-writer causal session and complete logit audit."""

    def __init__(self, ground, host: Callable, *, approved_hosts: set):
        self.ground = _check_ground(dict(ground))
        if not callable(host) or self.ground['observer_host'] not in approved_hosts:
            raise ValueError('Unapproved operational observer host')
        self.host = host
        self.tokens = []
        self.prompt_length = 0
        self.session_id = None
        self.logits_digest = None
        self.transcript_head = '0' * 64
        self.positions = []
        self.remote_calls = 0
        self.remote_weight_bytes_received = 0  # protocol never accepts weights
        self.epoch = 0
        self.owner = 'local'

    def _call(self, operation, arguments):
        if operation not in _OPERATIONS:
            raise ValueError('Unadmitted operation')
        req = {'schema': CALL_SCHEMA, 'operation': operation,
               'ground_digest': self.ground['ground_digest'],
               'model': self.ground['model'], 'revision': self.ground['revision'],
               'arguments': arguments}
        reply = self.host(req)
        if not isinstance(reply,dict) or (reply.get('schema') != RESULT_SCHEMA or
            reply.get('operation') != operation or
            reply.get('ground_digest') != self.ground['ground_digest'] or
            reply.get('model') != self.ground['model'] or
            reply.get('revision') != self.ground['revision'] or
            reply.get('observer_host') != self.ground['observer_host'] or
            reply.get('status') != 'HOST_OPERATION_COMPLETED'):
            raise ValueError('Unqualified operational host result')
        result = reply.get('result')
        if (not isinstance(result,dict) or
                reply.get('result_digest') != digest(RESULT_SCHEMA, result)):
            raise ValueError('Unqualified operational result digest')
        self.remote_calls += 1
        return result

    def encode(self, text):
        if not isinstance(text,str) or not 1 <= len(text.encode('utf-8')) <= MAX_TEXT_BYTES:
            raise ValueError('Invalid text intention extent')
        result = self._call('encode', {'text': text})
        tokens = result.get('tokens')
        _check_tokens(tokens, self.ground['vocab_size'])
        return tokens

    def decode(self, tokens):
        _check_tokens(tokens,self.ground['vocab_size'],allow_empty=True)
        result = self._call('decode', {'tokens': tokens})
        if (not isinstance(result.get('text'),str) or
                len(result['text'].encode('utf-8')) > MAX_TEXT_BYTES):
            raise ValueError('Invalid decoded text')
        return result['text']

    def open_tokens(self,tokens,*,prompt_length=None):
        _check_tokens(tokens,self.ground['vocab_size'])
        if len(tokens) >= self.ground['max_context']:
            raise ValueError('No causal room for another token')
        result = self._call('open', {'tokens': tokens})
        if (not isinstance(result.get('session_id'),str) or
                not 1 <= len(result['session_id']) <= 200 or
                result.get('prefix_digest') != prefix_digest(tokens) or
                not isinstance(result.get('logits_sha256'),str) or
                not _HEX.fullmatch(result['logits_sha256']) or
                result.get('token_count') != len(tokens)):
            raise ValueError('Host opened a different causal prefix')
        self.tokens = list(tokens)
        if prompt_length is not None and (type(prompt_length) is not int or not 1 <= prompt_length <= len(tokens)):
            raise ValueError('Invalid original prompt length')
        self.prompt_length = len(tokens) if prompt_length is None else prompt_length
        self.session_id = result['session_id']
        self.logits_digest = result['logits_sha256']
        return result

    def open_text(self,text):
        return self.open_tokens(self.encode(text))

    def project_next_token(self,*,tile_size=MAX_LOGITS_PER_TILE):
        """Stream all vocabulary logits as <=64KiB observations; verify argmax.

        The model forward pass and weights stay at the host. VL64 verifies
        complete tile coverage and byte-consistent logits supplied by that host.
        """
        if self.session_id is None:
            raise ValueError('No admitted model session')
        if type(tile_size) is not int or not 1 <= tile_size <= MAX_LOGITS_PER_TILE:
            raise ValueError('Invalid bounded logit tile')
        count = self.ground['vocab_size']
        computed=hashlib.sha256()
        maximum=-float('inf'); chosen=-1; observations=[]
        before=self.remote_calls
        for start in range(0,count,tile_size):
            take=min(tile_size,count-start)
            result=self._call('logits_tile',{
                'session_id':self.session_id,'prefix_digest':prefix_digest(self.tokens),
                'start':start,'count':take})
            if (result.get('start') != start or result.get('count') != take or
                    result.get('session_id') != self.session_id or
                    result.get('prefix_digest') != prefix_digest(self.tokens) or
                    result.get('full_logits_sha256') != self.logits_digest):
                raise ValueError('Missing/misaddressed model logit tile')
            raw_hex=result.get('f32le_hex')
            if (not isinstance(raw_hex,str) or len(raw_hex)!=8*take or
                    len(raw_hex)>MAX_FRAME_BYTES*2):
                raise ValueError('Oversized or incomplete logit tile')
            try: raw=bytes.fromhex(raw_hex)
            except ValueError as exc: raise ValueError('Nonhex logit tile') from exc
            if result.get('sha256')!=hashlib.sha256(raw).hexdigest():
                raise ValueError('Altered logit tile')
            computed.update(raw)
            for j,(z,) in enumerate(struct.iter_unpack('<f',raw)):
                if not math.isfinite(z):
                    raise ValueError('Nonfinite model logits')
                index=start+j
                if z>maximum: maximum=z;chosen=index
            observations.append(result['sha256'])
        if computed.hexdigest()!=self.logits_digest or chosen<0:
            raise ValueError('Host full-logit commitment differs from observed tiles')
        return {'token': chosen,'maximum_logit': maximum,
                'logits_sha256':computed.hexdigest(),'tile_sha256':observations,
                'logit_tiles':len(observations),'host_calls':self.remote_calls-before}

    def advance(self, *,tile_size=MAX_LOGITS_PER_TILE):
        if len(self.tokens) >= self.ground['max_context']:
            raise ValueError('Causal context exhausted')
        projection=self.project_next_token(tile_size=tile_size)
        new=self.tokens+[projection['token']]
        result=self._call('advance', {'session_id':self.session_id,
                    'old_prefix_digest':prefix_digest(self.tokens),
                    'chosen_token':projection['token'],
                    'observed_logits_sha256':projection['logits_sha256']})
        if (result.get('prefix_digest')!=prefix_digest(new) or
                result.get('token_count')!=len(new) or
                not isinstance(result.get('session_id'),str) or
                not 1<=len(result['session_id'])<=200 or
                not isinstance(result.get('logits_sha256'),str) or
                not _HEX.fullmatch(result['logits_sha256'])):
            raise ValueError('Host committed an inconsistent causal transition')
        transition = {'previous':self.transcript_head,
                      'previous_prefix':prefix_digest(self.tokens),
                      'new_prefix':prefix_digest(new),
                      'chosen_token':projection['token'],
                      'observed_logits_sha256':projection['logits_sha256'],
                      'logit_tiles':projection['tile_sha256']}
        # Atomic publication of the logical token only after all requests pass.
        self.transcript_head=digest('vl64.operational-token-transition.v1',transition)
        self.positions.append({'position':len(self.tokens),
            'token':projection['token'],'logit_tiles':projection['logit_tiles'],
            'transcript_head':self.transcript_head})
        self.tokens=new
        self.session_id=result['session_id']
        self.logits_digest=result['logits_sha256']
        return projection['token']

    def generate(self,text,*,max_new_tokens=16,stops=None,tile_size=MAX_LOGITS_PER_TILE):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=256:
            raise ValueError('Unbounded generation intention')
        if stops is None:stops=[]
        _check_tokens(stops,self.ground['vocab_size'],allow_empty=True)
        self.open_text(text)
        generated=[]
        for _ in range(max_new_tokens):
            token=self.advance(tile_size=tile_size)
            generated.append(token)
            if token in stops: break
        return {'status':'HOST_RESIDENT_MODEL_TOKENS_EXECUTED',
            'scope':'host_computed_logits; vl64_verified_complete_argmax_not_remote_math',
            'ground_digest':self.ground['ground_digest'],
            'input_token_count':self.prompt_length,
            'generated_tokens':generated,'generated_text':self.decode(generated),
            'remote_weight_bytes_received':self.remote_weight_bytes_received,
            'transcript_head':self.transcript_head,'host_calls':self.remote_calls,
            'position_receipts':list(self.positions)}

    def checkpoint(self):
        if self.session_id is None:raise ValueError('No causal session to checkpoint')
        record={'schema':CHECKPOINT_SCHEMA,'ground_digest':self.ground['ground_digest'],
                'tokens':list(self.tokens),'prompt_length':self.prompt_length,
                'prefix_digest':prefix_digest(self.tokens),
                'logits_sha256':self.logits_digest,
                'transcript_head':self.transcript_head,
                'owner':self.owner,'epoch':self.epoch}
        record['checkpoint_digest']=digest(CHECKPOINT_SCHEMA,record)
        return record

    @classmethod
    def restore(cls, ground, host, *, approved_hosts, checkpoint,
                expected_checkpoint_digest, owner, epoch):
        _check_ground(ground)
        if (not isinstance(checkpoint,dict) or
                set(checkpoint)!={'schema','ground_digest','tokens','prompt_length',
                     'prefix_digest','logits_sha256','transcript_head','owner',
                     'epoch','checkpoint_digest'} or
                checkpoint.get('schema') != CHECKPOINT_SCHEMA or
                checkpoint.get('ground_digest') != ground['ground_digest'] or
                checkpoint.get('checkpoint_digest') != expected_checkpoint_digest or
                checkpoint['checkpoint_digest'] != digest(CHECKPOINT_SCHEMA,
                    {k:v for k,v in checkpoint.items() if k!='checkpoint_digest'}) or
                type(checkpoint.get('epoch')) is not int or type(epoch) is not int or
                epoch <= checkpoint['epoch'] or
                not isinstance(owner,str) or not owner or
                type(checkpoint.get('prompt_length')) is not int or
                not 1 <= checkpoint['prompt_length'] <= len(checkpoint['tokens']) or
                not isinstance(checkpoint.get('transcript_head'),str) or
                not _HEX.fullmatch(checkpoint['transcript_head'])):
            raise ValueError('Untrusted, stale or incomplete model-orbit checkpoint')
        nav=cls(ground,host,approved_hosts=approved_hosts)
        nav.open_tokens(checkpoint['tokens'],prompt_length=checkpoint['prompt_length'])
        if (checkpoint['prefix_digest']!=prefix_digest(nav.tokens) or
                checkpoint['logits_sha256']!=nav.logits_digest):
            raise ValueError('Replacement host cannot reproduce exact causal logits')
        nav.transcript_head=checkpoint['transcript_head']
        nav.owner,nav.epoch=owner,epoch
        return nav


    def transfer_to(self, new_host, *, new_approved_hosts, expected_checkpoint_digest,
                    owner, epoch):
        """Rebind a live token prefix to a new observer Ground via a strict challenge.

        This is evidence that *one prefix* gives the exact same F32 logit bits
        under the new host. It does not establish identical trained parameters
        for all possible inputs and is not remote attestation.
        """
        checkpoint=self.checkpoint()
        if checkpoint['checkpoint_digest']!=expected_checkpoint_digest:
            raise ValueError('External checkpoint high-water pin differs')
        new_ground=observe_operational_ground(
            new_host,expected_model=self.ground['model'],
            expected_revision=self.ground['revision'],
            expected_artifact=self.ground['model_artifact_identity'],
            approved_hosts=new_approved_hosts)
        new=OperationalModelNavigator(new_ground,new_host,
                approved_hosts=new_approved_hosts)
        if type(epoch) is not int or epoch<=checkpoint['epoch']:
            raise ValueError('Stale reassignment epoch')
        new.open_tokens(checkpoint['tokens'],prompt_length=checkpoint['prompt_length'])
        if (new.logits_digest!=checkpoint['logits_sha256'] or
                prefix_digest(new.tokens)!=checkpoint['prefix_digest']):
            raise ValueError('Replacement observer failed causal output challenge')
        new.transcript_head=checkpoint['transcript_head']
        new.owner,new.epoch=owner,epoch
        record={'schema':'vl64.operational-host-reassignment.v1',
                'source_ground':self.ground['ground_digest'],
                'replacement_ground':new_ground['ground_digest'],
                'checkpoint_digest':checkpoint['checkpoint_digest'],
                'prefix_digest':checkpoint['prefix_digest'],
                'independent_host_prefix_logit_match':True,
                'observed_logits_sha256':new.logits_digest,
                'owner':owner,'epoch':epoch,
                'scope':'same_prefix_F32_match_not_whole_model_equivalence_or_authentication'}
        record['reassignment_digest']=digest(record['schema'],record)
        return new,record


class BoundedTaskAgent:
    """Use model-produced strict JSON to request only caller-approved tools.

    No shell, filesystem, internet or other actions exist without the caller
    registering them. Model text is untrusted and never grants tool authority.
    """
    def __init__(self,navigator_factory,tools,*,max_rounds=3,max_tokens=128):
        if (not callable(navigator_factory) or not isinstance(tools,dict) or
                any(not isinstance(k,str) or not _ID.fullmatch(k) or
                    not callable(v) for k,v in tools.items()) or
                type(max_rounds) is not int or not 1<=max_rounds<=8 or
                type(max_tokens) is not int or not 1<=max_tokens<=256):
            raise ValueError('Invalid bounded model agent policy')
        self.navigator_factory,self.tools=navigator_factory,dict(tools)
        self.max_rounds,self.max_tokens=max_rounds,max_tokens

    def run(self,task):
        if not isinstance(task,str) or not 1<=len(task.encode())<=4096:
            raise ValueError('Invalid model task')
        context=task;events=[]
        for round_idx in range(self.max_rounds):
            nav=self.navigator_factory()
            response=nav.generate(context,max_new_tokens=self.max_tokens)
            try: obj=json.loads(response['generated_text'].strip())
            except (ValueError,TypeError) as exc:
                raise ValueError('Model output is not a structured tool/final decision') from exc
            if not isinstance(obj,dict):raise ValueError('Unstructured model decision')
            if set(obj)=={'final'} and isinstance(obj['final'],str):
                return {'status':'BOUNDED_MODEL_TASK_COMPLETED','final':obj['final'],
                        'events':events,'rounds':round_idx+1,
                        'scope':'model_decisions_host_claimed; tools_caller_authorized'}
            if (set(obj)!={'tool','arguments'} or not isinstance(obj['tool'],str) or
                    obj['tool'] not in self.tools or not isinstance(obj['arguments'],dict)):
                raise ValueError('Model attempted an unapproved or malformed tool action')
            args=obj['arguments']
            if len(canonical(args))>8192:
                raise ValueError('Tool argument budget exceeded')
            answer=self.tools[obj['tool']](**args)
            result={'tool':obj['tool'],'arguments':args,'result':answer}
            if len(canonical(result))>8192:
                raise ValueError('Tool result budget exceeded')
            events.append({'tool':obj['tool'],'result_digest':digest('vl64.agent-tool-result.v1',result)})
            context=(task+'\nObserved tool result: '+canonical(result).decode()+
                     '\nReturn a JSON object with final or an authorized tool request.')
            if len(context.encode())>MAX_TEXT_BYTES:
                raise ValueError('Model task context budget exceeded')
        return {'status':'TASK_ROUND_BUDGET_REACHED','events':events,'rounds':self.max_rounds}
