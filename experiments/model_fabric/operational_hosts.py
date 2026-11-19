"""Host-resident trained-language and optional neural transformer implementations.

The virtual layer imports none of these classes. Model weights are owned by
hosts; the VL64 navigator exchanges text, token IDs, F32 logit tiles and
causal receipts only. This module is a local transport fixture/host adapter,
not an authenticated remote network transport or a new model architecture.
"""
from __future__ import annotations

import hashlib
import math
import struct
from .operational_ground import (
    GROUND_SCHEMA, CALL_SCHEMA, RESULT_SCHEMA, MAX_LOGITS_PER_TILE,
    MAX_TOKENS, _OPERATIONS, _check_tokens, digest, prefix_digest,
)


class OperationalHostBase:
    def __init__(self, *, model, revision, observer_host,
                 artifact_identity, tokenizer_id, vocab_size, max_context):
        self.description={
            'schema':GROUND_SCHEMA,'model':model,'revision':revision,
            'observer_host':observer_host,'model_artifact_identity':artifact_identity,
            'tokenizer_id':tokenizer_id,'vocab_size':vocab_size,
            'max_context':min(max_context,MAX_TOKENS),
            'operations':list(_OPERATIONS),
            'trust_scope':'host_capability_observation_not_remote_weight_or_execution_attestation'}
        self.bound_ground=digest(GROUND_SCHEMA,self.description)
        self.sessions={}

    def encode(self,text):raise NotImplementedError
    def decode(self,tokens):raise NotImplementedError
    def calculate_logits(self,tokens):raise NotImplementedError

    def _open(self,tokens):
        _check_tokens(tokens,self.description['vocab_size'])
        if len(tokens)>=self.description['max_context']:
            raise ValueError('Host causal context exhausted')
        # The host owns and prepares the trained operation. Only its output is
        # projected into the Virtual Layer; the model weights never are.
        values=self.calculate_logits(tokens)
        if len(values)!=self.description['vocab_size'] or not all(math.isfinite(x) for x in values):
            raise ValueError('Incomplete/nonfinite host trained model output')
        raw=struct.pack('<'+'f'*len(values),*values)
        lid=hashlib.sha256(raw).hexdigest()
        p=prefix_digest(tokens)
        session=digest('vl64.host-model-session.v1',{'prefix':p,'logits':lid})
        self.sessions[session]=(list(tokens),raw,lid)
        # Bounded host memory across sessions; no need to retain all prefixes.
        if len(self.sessions)>8:
            old=next(iter(self.sessions));self.sessions.pop(old,None)
        return {'session_id':session,'prefix_digest':p,
                'logits_sha256':lid,'token_count':len(tokens)}

    def __call__(self,request):
        if not isinstance(request,dict) or request.get('schema')!=CALL_SCHEMA:
            raise ValueError('Unadmitted host request schema')
        op=request.get('operation')
        if op=='describe':return dict(self.description)
        if (op not in _OPERATIONS or request.get('model')!=self.description['model'] or
                request.get('revision')!=self.description['revision'] or
                request.get('ground_digest')!=self.bound_ground or
                not isinstance(request.get('arguments'),dict)):
            raise ValueError('Unbound or unknown host operation')
        args=request['arguments']
        if op=='encode':
            if set(args)!={'text'} or not isinstance(args['text'],str):raise ValueError('Unadmitted text')
            result={'tokens':self.encode(args['text'])}
        elif op=='decode':
            if set(args)!={'tokens'}:raise ValueError('Unadmitted decode request')
            _check_tokens(args['tokens'],self.description['vocab_size'],allow_empty=True)
            result={'text':self.decode(args['tokens'])}
        elif op=='open':
            if set(args)!={'tokens'}:raise ValueError('Unadmitted open request')
            result=self._open(args['tokens'])
        elif op=='logits_tile':
            if set(args)!={'session_id','prefix_digest','start','count'}:
                raise ValueError('Unadmitted logit tile request')
            sid=args['session_id'];session=self.sessions.get(sid)
            first,n=args['start'],args['count']
            if (session is None or args['prefix_digest']!=prefix_digest(session[0]) or
                    type(first) is not int or type(n) is not int or
                    first<0 or n<1 or n>MAX_LOGITS_PER_TILE or
                    first+n>self.description['vocab_size']):
                raise ValueError('Unadmitted host logit extent')
            tile=session[1][first*4:(first+n)*4]
            result={'session_id':sid,'prefix_digest':args['prefix_digest'],
                    'start':first,'count':n,'f32le_hex':tile.hex(),
                    'sha256':hashlib.sha256(tile).hexdigest(),
                    'full_logits_sha256':session[2]}
        else:
            if set(args)!={'session_id','old_prefix_digest','chosen_token','observed_logits_sha256'}:
                raise ValueError('Unadmitted causal advancement')
            sid=args['session_id'];session=self.sessions.get(sid)
            t=args['chosen_token']
            if (session is None or args['old_prefix_digest']!=prefix_digest(session[0]) or
                    args['observed_logits_sha256']!=session[2] or
                    type(t) is not int or not 0<=t<self.description['vocab_size'] or
                    len(session[0])>=self.description['max_context']-1):
                raise ValueError('Host refused causal state advancement')
            result=self._open(session[0]+[t])
        return {'schema':RESULT_SCHEMA,'operation':op,
                'ground_digest':self.bound_ground,'model':self.description['model'],
                'revision':self.description['revision'],
                'observer_host':self.description['observer_host'],
                'status':'HOST_OPERATION_COMPLETED','result':result,
                'result_digest':digest(RESULT_SCHEMA,result)}


class HostedTrainedByteLanguage(OperationalHostBase):
    """Actual count-trained byte LM: trained parameters remain in this host."""
    def __init__(self, trained_ground, *, observer_host='local-byte-model-host'):
        from .byte_lm import ByteLanguageGround
        model=ByteLanguageGround(trained_ground)
        self.trained=model
        sha=trained_ground['ground_digest']
        super().__init__(model='vl64/trained-byte-language',revision=sha,
            observer_host=observer_host,artifact_identity=sha,
            tokenizer_id='identity_utf8_byte_v1',vocab_size=256,max_context=4096)

    def encode(self,text):return list(text.encode('utf-8'))
    def decode(self,tokens):return bytes(tokens).decode('utf-8','replace')
    def calculate_logits(self,tokens):
        probs=self.trained.probabilities(tokens[-1])
        return [math.log(p) for p in probs]


class HostedTransformersLanguage(OperationalHostBase):
    """Optional real Hugging Face autoregressive model host (PyTorch backend).

    Loads model weights inside the host only; the VL model Ground retains no
    parameter bytes. CPU FP32 ensures deterministic-reference opportunities.
    Some architectures need specialized runtimes; unsupported models refuse.
    Install torch and transformers separately; no core dependency added.
    """
    def __init__(self, model_name, revision, *, observer_host='torch-model-host',
                 max_context=512, local_files_only=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Require pinned immutable model commit for transformer host')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,
                              trust_remote_code=False,local_files_only=local_files_only)
        self.model=AutoModelForCausalLM.from_pretrained(model_name,revision=revision,
                              trust_remote_code=False,local_files_only=local_files_only,
                              torch_dtype=torch.float32)
        self.model.eval()
        vocab=int(self.model.config.vocab_size)
        context=int(getattr(self.model.config,'max_position_embeddings',max_context))
        super().__init__(model=model_name,revision=revision,
            observer_host=observer_host,
            artifact_identity=digest('vl64.host-model-revision-claim.v1',
                                     {'model':model_name,'revision':revision}),
            tokenizer_id='hf_tokenizer_pinned_v1',vocab_size=vocab,
            max_context=min(max_context,context))

    def encode(self,text):
        return list(self.tokenizer.encode(text,add_special_tokens=False))
    def decode(self,tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens=True)
    def calculate_logits(self,tokens):
        """Reuse host-owned KV only for the exact next causal prefix.

        Branches, restored sessions and modified prefixes force a fresh prefill.
        Never place KV/weight data in the Virtual Layer or its Ground receipt.
        """
        previous=getattr(self,'_cached_prefix',None)
        past=getattr(self,'_cached_kv',None)
        incremental=past is not None and previous==tuple(tokens[:-1])
        payload=tokens[-1:] if incremental else tokens
        kwargs={'input_ids':self.torch.tensor([payload],dtype=self.torch.long),
                'use_cache':True}
        if incremental:kwargs['past_key_values']=past
        try:
            with self.torch.inference_mode():
                result=self.model(**kwargs)
                values=result.logits[0,-1].float().cpu().tolist()
            if (len(values)!=self.description['vocab_size'] or
                    not all(math.isfinite(x) for x in values)):
                raise ValueError('Incomplete/nonfinite cached model output')
        except Exception:
            # Failed decoder steps cannot leave an apparently valid causal head.
            self._cached_prefix=None;self._cached_kv=None
            raise
        self._cached_prefix=tuple(tokens)
        self._cached_kv=getattr(result,'past_key_values',None)
        return values
