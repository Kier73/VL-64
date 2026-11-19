"""Host-side causal KV incremental reuse, with no weights entering VL64."""
import contextlib
import pytest

from experiments.model_fabric.operational_hosts import HostedTransformersLanguage


class FakeTorch:
    long='long'
    def tensor(self,rows,dtype):
        assert dtype==self.long
        return rows
    def inference_mode(self):return contextlib.nullcontext()


class FakeVector:
    def __init__(self,vals):self.values=vals
    def float(self):return self
    def cpu(self):return self
    def tolist(self):return list(self.values)


class FakeLogits:
    def __init__(self,vals):self.v=vals
    def __getitem__(self,key):
        assert key==(0,-1)
        return FakeVector(self.v)


class FakeDecoder:
    def __init__(self):self.calls=[];self.fail=False;self.cache_enabled=True
    def __call__(self,*,input_ids,use_cache,past_key_values=None):
        assert use_cache
        before=() if past_key_values is None else past_key_values
        now=tuple(input_ids[0]);all_tokens=before+now
        self.calls.append({'received':now,'previous':before,'full_prefix':all_tokens})
        if self.fail:raise RuntimeError('simulated model compute failure')
        # Match a full-sequence mathematical function at every causal prefix.
        logits=[float(-abs(sum(all_tokens)-i)) for i in range(20)]
        class Result:pass
        r=Result();r.logits=FakeLogits(logits)
        r.past_key_values=all_tokens if self.cache_enabled else None
        return r


def make_host(model=None):
    obj=HostedTransformersLanguage.__new__(HostedTransformersLanguage)
    obj.torch=FakeTorch()
    obj.model=model if model is not None else FakeDecoder()
    obj.description={'vocab_size':20}
    return obj


def test_exact_causal_prefix_uses_one_token_only_after_prefill():
    host=make_host()
    a=host.calculate_logits([2,3,4])
    b=host.calculate_logits([2,3,4,5])
    c=host.calculate_logits([2,3,4,5,6])
    assert [len(e['received']) for e in host.model.calls]==[3,1,1]
    expected=[float(-abs(20-i)) for i in range(20)]
    assert c==expected
    assert host._cached_prefix==(2,3,4,5,6)
    assert a!=b and b!=c


def test_sibling_prefix_forces_full_prefill():
    host=make_host()
    host.calculate_logits([2,3,4]);host.calculate_logits([2,3,4,5])
    branch=host.calculate_logits([2,3,9])
    assert host.model.calls[-1]['received']==(2,3,9)
    assert branch==[float(-abs(14-i)) for i in range(20)]


def test_failure_fences_cache_and_requires_new_prefill():
    host=make_host()
    host.calculate_logits([1,2,3])
    host.model.fail=True
    with pytest.raises(RuntimeError,match='failure'):
        host.calculate_logits([1,2,3,4])
    assert host._cached_kv is None
    host.model.fail=False
    host.calculate_logits([1,2,3,4])
    assert len(host.model.calls[-1]['received'])==4


def test_missing_kv_response_recomputes_instead_of_faking_incrementality():
    model=FakeDecoder();model.cache_enabled=False
    host=make_host(model)
    host.calculate_logits([1,2,3])
    host.calculate_logits([1,2,3,4])
    assert [len(e['received']) for e in model.calls]==[3,4]


def test_nonfinite_logits_refused_before_committing_cache():
    host=make_host()
    def corrupted(**kwargs):
        class Result:pass
        r=Result();r.logits=FakeLogits([float('nan')]*20)
        r.past_key_values=(1,2)
        return r
    host.model=corrupted
    with pytest.raises(ValueError,match='nonfinite'):
        host.calculate_logits([1,2])
    assert host._cached_kv is None
