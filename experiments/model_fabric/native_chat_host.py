"""Opt-in tokenizer bridge: the model host, not VL64, owns its chat template.

A distinct tokenizer identity makes native-chat and text-role prompts impossible
to silently substitute for one another in an admitted operational Ground.
"""
from __future__ import annotations

import json
from .operational_ground import MAX_TOKENS, digest
from .operational_hosts import HostedTransformersLanguage
from .intention_fabric import CHAT_PREFIX, _pairs, _messages


class HostedNativeChatTransformer(HostedTransformersLanguage):
    """Chat-template adapter for supported instruction-tuned HF model hosts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(getattr(self.tokenizer, 'apply_chat_template', None)):
            raise ValueError('Tokenizer has no native chat template operation')
        self.description['tokenizer_id']='hf_native_chat_template_v1'
        self.bound_ground=digest(self.description['schema'],self.description)

    def encode(self, text):
        if not isinstance(text,str):raise ValueError('Malformed tokenizer request')
        if not text.startswith(CHAT_PREFIX):
            raise ValueError('Native chat host refuses substitute plain-text tokenization')
        try:
            envelope=json.loads(text[len(CHAT_PREFIX):],object_pairs_hook=_pairs,
               parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite chat record')))
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid native-chat envelope') from exc
        if (not isinstance(envelope,dict) or set(envelope)!={
            'schema','messages','add_generation_prompt'} or
            envelope['schema']!='vl64.role-messages.v1' or
            envelope['add_generation_prompt'] is not True):
            raise ValueError('Invalid native-chat template contract')
        roles=_messages(envelope['messages'])
        tokens=self.tokenizer.apply_chat_template(roles,tokenize=True,
                add_generation_prompt=True)
        if (not isinstance(tokens,list) or not 1<=len(tokens)<min(
                self.description['max_context'],MAX_TOKENS) or
                any(type(t) is not int or not 0<=t<self.description['vocab_size']
                    for t in tokens)):
            raise ValueError('Native chat template returned unadmitted token IDs')
        return tokens
