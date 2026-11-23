"""Regression controls for the head pullback and late-bound token projections."""
import hashlib
import math
import struct
from itertools import product

import pytest

from experiments.model_fabric import addressed_delta as runtime
from experiments.model_fabric.architecture_ops import (
    depthwise_causal_conv_step, qwen_gated_delta_step, qwen_gated_rms_norm,
)


def test_pullback_non_square_state():
    """For S ∈ R^(2×3), compare Sᵀq to reverse-mode evaluation on every binary length-4 prefix."""
    k, v = 2, 3
    bank = {t: [math.sin(i+.3*t) for i in range(2*k+2*v)] + [.2+.2*t, .9-.1*t] for t in [0,1]}
    conv = [[.6, -.2, .7] for _ in range(2*k+v)]
    gain, row = [1.,2.,.5], [.1,-.3,1.]
    g = dict(key_width=k,value_width=v,eps=1e-6,context=4)
    program = g,conv,gain,row,bank
    for tokens in product((0,1),repeat=4):
        hist, state = [], None
        for t in tokens:
            entry = bank[t]
            m,hist = depthwise_causal_conv_step(entry[:7],conv,hist)
            core,state = qwen_gated_delta_step([m[:k]],[m[k:2*k]],[m[2*k:]],
                [math.log(entry[-1])],[entry[-2]],state)
        expected = qwen_gated_rms_norm(core[0],entry[7:10],gain)
        actual = runtime.evaluate(program,tokens)
        assert actual["head_output"] == pytest.approx(expected,abs=1e-13)
        assert actual["row_contribution"] == pytest.approx(sum(x*y for x,y in zip(expected,row)),abs=1e-13)
    for tokens in ([], [2], [1]*5):
        with pytest.raises(ValueError):
            runtime.evaluate(program,tokens)


def test_resolver_publishes_only_complete_token_entries(monkeypatch):
    """Φ(t) becomes visible only after every verified W_j·x_t is available; Φ(t) is reusable."""
    blob = struct.pack("<4f",.5,1,.25,2)
    part = dict(dtype="F32",offset=0,size=4,file_size=16,shard="w.safetensors",
                sha256=hashlib.sha256(blob[:4]).hexdigest())
    address = dict(part,hidden=1,vocab=2)
    address.pop("sha256")
    remote = dict(embedding=address,norm=[part],qkv=[part]*3,z=[part],a=[part],b=[dict(part,sha256="0"*64)])
    bank = {}
    g = dict(remote=remote,model="a/b",revision="1"*40,context=3,eps=1e-6,a_log=0.,dt_bias=0.)
    program = (g,None,None,None,bank)

    def fetch(url,start,length):
        """S[a:a+n] uses half-open byte coordinates on the pinned object."""
        return blob[start:start+length],len(blob)

    monkeypatch.setattr(runtime,"range_bytes",fetch)
    with pytest.raises(ValueError,match="diverged"):
        runtime.resolve(program,[1,1])
    assert not bank
    remote["b"] = [part]
    assert runtime.resolve(program,[1,1]) == 32
    assert runtime.resolve(program,[1]) == 0
    assert list(bank) == [1]
