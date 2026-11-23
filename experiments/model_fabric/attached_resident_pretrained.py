"""MPF-003: attach an already-resident original pretrained model to VL-64.

No from_pretrained, checkpoint loader, weight-range reader, filesystem fetch,
or model download exists here. The source host supplies existing model, tokenizer
and compute runtime objects. Identity is an externally checked admission pin,
not an intrinsic proof that those objects match a claimed checkpoint.
"""
from __future__ import annotations

import re

from .operational_ground import GROUND_SCHEMA, MAX_TOKENS, digest, _check_tokens
from .operational_hosts import OperationalHostBase, HostedTransformersLanguage

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class AttachedResidentPretrainedLanguage(HostedTransformersLanguage):
    """Wrap host-owned live transformer state without loading or copying it.

    The host's model object, tokenizer, torch instance, and optional KV cache
    remain outside the VL-64 observer.  Transformer arithmetic is inherited
    from the original HostedTransformersLanguage.calculate_logits().
    """

    def __init__(self, *, model, tokenizer, torch, repository: str, revision: str,
                 artifact_identity: str, observer_host: str,
                 native_chat: bool = False, max_context: int = 512):
        if (not isinstance(repository, str) or "/" not in repository
                or not isinstance(revision, str)
                or _REVISION.fullmatch(revision) is None
                or not isinstance(artifact_identity, str)
                or _SHA.fullmatch(artifact_identity) is None
                or not isinstance(observer_host, str) or not observer_host
                or type(native_chat) is not bool
                or type(max_context) is not int
                or not 2 <= max_context <= MAX_TOKENS):
            raise ValueError("Preloaded model requires exact source and artifact pins")
        if (not callable(getattr(model, "eval", None))
                or not callable(getattr(model, "__call__", None))
                or not callable(getattr(tokenizer, "encode", None))
                or not callable(getattr(tokenizer, "decode", None))
                or not callable(getattr(torch, "tensor", None))
                or not callable(getattr(torch, "inference_mode", None))
                or not hasattr(torch, "long")
                or (native_chat and not callable(
                    getattr(tokenizer, "apply_chat_template", None)))):
            raise ValueError("Source host must already supply executable model and tokenizer")
        config = getattr(model, "config", None)
        vocab = getattr(config, "vocab_size", None)
        maximum = getattr(config, "max_position_embeddings", None)
        if (type(vocab) is not int or not 2 <= vocab <= 10_000_000
                or (maximum is not None and
                    (type(maximum) is not int or maximum < 2))):
            raise ValueError("Resident transformer configuration is not admissible")
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.native_chat = native_chat
        self._cached_prefix = None
        self._cached_kv = None
        model.eval()  # no instantiation or weight copying; host owns this object
        OperationalHostBase.__init__(
            self, model=repository, revision=revision, observer_host=observer_host,
            artifact_identity=artifact_identity,
            tokenizer_id=("hf_native_chat_template_v1" if native_chat
                          else "hf_tokenizer_pinned_v1"),
            vocab_size=vocab,
            max_context=min(max_context, maximum or max_context),
        )

    def encode(self, text):
        if self.native_chat:
            # Reuse the original host-side chat-template implementation; never
            # hard-code tokenizer IDs into a virtual observer.
            from .native_chat_host import HostedNativeChatTransformer
            tokens = HostedNativeChatTransformer.encode(self, text)
        else:
            tokens = super().encode(text)
        _check_tokens(tokens, self.description["vocab_size"])
        return tokens
