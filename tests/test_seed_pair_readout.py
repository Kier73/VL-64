"""Source-backed checks of sign(aᵀu+(Dᵀa)ᵀv) against the full vocabulary head."""
import json
import math

import pytest

from experiments.model_fabric import execution_dag
from experiments.model_fabric.execution_dag import compile_decoder_dag, compile_pair_readout, execute_tokens
from test_execution_dag import qualified_host, scalar_reference, synthetic_pretrained_ground
from vld import domain_digest


HOSTS = {"local-f32-observer"}


def counted_host(ground, data):
    """Track R=|requests| and B=Σ(end−begin) independently of the executor."""
    host, requests = qualified_host(ground, data), []

    def observe(request):
        """Append the next interval [begin,end) before forwarding to the qualified host."""
        requests.append(dict(request))
        return host(request)

    return observe, requests


def cost(requests):
    """Transferred extent B=Σₖ(eₖ−sₖ), in bytes."""
    return sum(r["byte_end_exclusive"]-r["byte_begin"] for r in requests)


@pytest.mark.parametrize("dtype,tied", [("F32", True), ("BF16", False)])
def test_pulled_pair_matches_full_and_stops_before_terminal_tail(dtype, tied):
    """For both legal head layouts, sign(L_i−L_j)=sign(aᵀu+bᵀv) on the tested inputs."""
    ground, data, weights = synthetic_pretrained_ground(dtype=dtype, tied=tied, layers=2)
    plan = compile_decoder_dag(ground)
    host, reads = counted_host(ground, data)
    pair = (3, 6)
    compiled = compile_pair_readout(plan, host, approved_hosts=HOSTS, pair=pair)
    down = "model.layers.1.mlp.down_proj.weight"
    head = "model.embed_tokens.weight" if tied else "lm_head.weight"
    assert [r["tensor"] for r in reads] == [head, head, "model.norm.weight", down]
    assert compiled["cold_weight_bytes_acquired"] == cost(reads)
    assert compiled["cold_exact_range_count"] == len(reads)
    spec = plan["tensors"]
    size = 4 if dtype=="F32" else 2
    h, w = spec[down]["shape"]
    assert cost(reads) == size*(3*h+h*w)
    compiled_pin = compiled["readout_digest"]
    cold = cost(reads)
    for tokens in ([7], [1, 7, 8]):
        reads.clear()
        full = execute_tokens(plan, tokens, host, approved_hosts=HOSTS)
        baseline = scalar_reference(tokens, weights, ground["architecture_fields"])
        assert full["status"] == "WEIGHT_BACKED_DECODER_FORWARD_EXECUTED"
        assert full["logits"] == pytest.approx(baseline, abs=2e-6)
        assert full["selected_next_token"] == baseline.index(max(baseline))
        full_reads = list(reads)
        reads.clear()
        pair_result = execute_tokens(plan, tokens, host, approved_hosts=HOSTS, pair_readout=compiled)
        # Terminal prefix omits exactly the final D, g and vocabulary head requests.
        assert reads == full_reads[:-3]
        assert [r["tensor"] for r in full_reads[-3:]] == [down, "model.norm.weight", head]
        assert pair_result["readout_digest"] == compiled_pin
        assert pair_result["cold_weight_bytes_acquired"] == cold
        assert pair_result["weight_bytes_acquired"] == cost(reads)
        assert pair_result["exact_range_count"] == len(reads)
        assert pair_result["ordered_pair"] == list(pair)
        assert pair_result["tie"] is False
        assert pair_result["selected_token"] == max(pair, key=baseline.__getitem__)
        assert (pair_result["score"]>0) == (baseline[pair[0]]>baseline[pair[1]])
        assert "logits" not in pair_result and "selected_next_token" not in pair_result
        body = {k:v for k,v in pair_result.items() if k!="receipt_digest"}
        assert pair_result["receipt_digest"] == domain_digest(body["schema"], body)


def test_equal_distinct_output_rows_yield_zero_tie():
    """w_i=w_j ⇒ a=b=0 ⇒ every residual yields a zero score and a pair tie."""
    ground, data, _ = synthetic_pretrained_ground(tied=False)
    plan = compile_decoder_dag(ground)
    pair = (4, 7)
    head = plan["tensors"]["lm_head.weight"]
    offset = head["data_offsets"][0]
    row_bytes = head["shape"][1]*4
    data = bytearray(data)
    left, right = (offset + t*row_bytes for t in pair)
    data[right:right+row_bytes] = data[left:left+row_bytes]
    host = qualified_host(ground, bytes(data))
    compiled = compile_pair_readout(plan, host, approved_hosts=HOSTS, pair=pair)
    full = execute_tokens(plan, [2], host, approved_hosts=HOSTS)
    assert full["logits"][4] == full["logits"][7]
    result = execute_tokens(plan, [2], host, approved_hosts=HOSTS, pair_readout=compiled)
    assert result["score"] == 0 and result["tie"] and result["selected_token"] is None


def test_bounded_transpose_compilation_crosses_row_and_range_boundaries(monkeypatch):
    """Under a range cap shorter than a D row, b_j still equals Σ_i D_ij (w₄ᵢ−w₇ᵢ)g_i."""
    monkeypatch.setattr(execution_dag, "MAX_RANGE", 32)
    ground, data, weights = synthetic_pretrained_ground()
    plan = compile_decoder_dag(ground)
    host, reads = counted_host(ground, data)
    compiled = compile_pair_readout(plan, host, approved_hosts=HOSTS, pair=[4, 7])
    (h,w), down = weights["model.layers.0.mlp.down_proj.weight"]
    _, head = weights["model.embed_tokens.weight"]
    _, gain = weights["model.norm.weight"]
    a = [(head[4*h+i]-head[7*h+i])*gain[i] for i in range(h)]
    b = [math.fsum(a[i]*down[i*w+j] for i in range(h)) for j in range(w)]
    assert compiled["a"] == a
    assert compiled["b"] == pytest.approx(b, rel=1e-14, abs=1e-16)
    assert max(r["byte_end_exclusive"]-r["byte_begin"] for r in reads) <= 32
    assert compiled["cold_exact_range_count"] == len(reads) > 4


def test_pair_identity_before_query_and_pinned_restore():
    """Require H(C)=external pin and C.plan=P before spending any query bytes."""
    ground, data, _ = synthetic_pretrained_ground()
    other, _, _ = synthetic_pretrained_ground(layers=2)
    plan, other_plan = compile_decoder_dag(ground), compile_decoder_dag(other)
    host, reads = counted_host(ground, data)
    for pair in ((2, 2), (0, 11), (-1, 2), (True, 3), (0, 1, 2)):
        with pytest.raises(ValueError):
            compile_pair_readout(plan, host, approved_hosts=HOSTS, pair=pair)
    assert not reads
    compiled = compile_pair_readout(plan, host, approved_hosts=HOSTS, pair=(1, 2))
    pin = compiled["readout_digest"]
    stored = json.loads(json.dumps(compiled))
    reads.clear()
    with pytest.raises(ValueError):
        execute_tokens(other_plan, [0], host, approved_hosts=HOSTS, pair_readout=compiled)
    with pytest.raises(ValueError):
        execute_tokens(plan, [0], host, approved_hosts=HOSTS, pair_readout=stored)
    assert not reads
    restored = execute_tokens(plan, [0], host, approved_hosts=HOSTS, pair_readout=stored,
                              expected_pair_readout_digest=pin)
    assert restored == execute_tokens(plan, [0], host, approved_hosts=HOSTS, pair_readout=compiled)
    reads.clear()
    for vector in ([], [float("nan")]*plan["architecture"]["hidden"], [1.0]*plan["architecture"]["hidden"]):
        altered = {**stored, "a":vector}
        with pytest.raises(ValueError):
            execute_tokens(plan, [0], host, approved_hosts=HOSTS, pair_readout=altered,
                           expected_pair_readout_digest=pin)
    assert not reads
