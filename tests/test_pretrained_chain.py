"""Independent two-layer scalar checks, handoff refusal, and observer routing."""
from __future__ import annotations

import copy
import hashlib
import math
import struct

import pytest

from vld import domain_digest
from experiments.model_fabric.execution_dag import compile_decoder_dag
from experiments.model_fabric.pretrained_chain import (
    HANDOFF_SCHEMA, GENESIS, admit_handoff, make_handoff,
    execute_pretrained_chain,
)
from test_execution_dag import synthetic_pretrained_ground, qualified_host, scalar_reference


def setup(*, dtype="BF16", tokens=(4, 3, 6), layers=2):
    observed, data, weights = synthetic_pretrained_ground(dtype=dtype, layers=layers)
    plan = compile_decoder_dag(observed)
    a = qualified_host(observed, data, observer="source-a")
    b = qualified_host(observed, data, observer="source-b")
    kwargs = dict(
        plan=plan, tokens=list(tokens), range_hosts={-1: a, 0: a, 1: b},
        approved_hosts={-1: {"source-a"}, 0: {"source-a"}, 1: {"source-b"}},
        compute_actors={0: "local-layer0"},
        approved_producers={"local-layer0"},
    )
    return kwargs, weights, observed["architecture_fields"], a, b


@pytest.mark.parametrize("dtype", ["BF16", "F32"])
@pytest.mark.parametrize("tokens", [[4], [4, 3, 6]])
def test_two_layers_match_independent_full_decoder_scalar_reference(dtype, tokens):
    kwargs, weights, arch, _, _ = setup(dtype=dtype, tokens=tokens)
    result = execute_pretrained_chain(**kwargs)
    vector = result["output"]
    gamma = weights["model.norm.weight"][1]
    scale = 1 / math.sqrt(math.fsum(v * v for v in vector) / len(vector) + arch["rms_norm_eps"])
    norm = [v * scale * g for v, g in zip(vector, gamma)]
    mshape, projection = weights["model.embed_tokens.weight"]
    logits = [math.fsum(projection[r * mshape[1] + j] * norm[j]
                        for j in range(mshape[1])) for r in range(mshape[0])]
    independently = scalar_reference(tokens, weights, arch)
    assert max(abs(x - y) for x, y in zip(logits, independently)) < 2e-6
    assert result["layers_executed"] == [0, 1]
    assert len(result["positions"]) == len(tokens)
    assert all(len(p["handoffs"]) == 1 for p in result["positions"])
    assert result["weight_payload_bytes_acquired"] > 0
    assert len(result["receipt_digest"]) == 64
    assert all(len(p["final_hidden"]["f32_sha256"]) == 64 for p in result["positions"])


def test_observer_reassignment_preserves_math_but_changes_weight_provenance():
    kwargs, _, _, a, b = setup()
    baseline = execute_pretrained_chain(**kwargs)
    moved = dict(kwargs)
    moved["range_hosts"] = {-1: b, 0: b, 1: a}
    moved["approved_hosts"] = {-1: {"source-b"}, 0: {"source-b"}, 1: {"source-a"}}
    rerun = execute_pretrained_chain(**moved)
    assert baseline["output"] == rerun["output"]
    assert baseline["range_chains"] != rerun["range_chains"]
    assert baseline["last_handoff_digest"] != rerun["last_handoff_digest"]
    assert rerun["tokens"] == baseline["tokens"]
    assert rerun["exact_range_count"] == baseline["exact_range_count"]


def test_reordered_or_changed_handoff_refused_even_with_recalculated_digest():
    kwargs, _, _, _, _ = setup()
    plan = kwargs["plan"]
    packet = make_handoff(plan, 0, 0, [0.125] * plan["architecture"]["hidden"],
                          producer="local-layer0", range_chain="a"*64)
    assert admit_handoff(plan, packet, index=0, position=0, preceding=GENESIS,
                         approved_producers={"local-layer0"},
                         expected_vector=[0.125]*8) == [0.125]*8
    for field, value in (("layer_from", 1), ("position", 1),
                         ("target_node", "layer.2.input_norm"),
                         ("producer", "malicious-host"),
                         ("predecessor_handoff_digest", "b"*64)):
        altered = dict(packet, **{field: value})
        altered["handoff_digest"] = domain_digest(
            HANDOFF_SCHEMA, {k:v for k,v in altered.items() if k not in ("payload","handoff_digest")})
        with pytest.raises(ValueError, match="Handoff violates"):
            admit_handoff(plan, altered, index=0, position=0, preceding=GENESIS,
                          approved_producers={"local-layer0"},
                          expected_vector=[0.125]*8)
    altered = dict(packet)
    altered["payload"] = b"\x00" * len(packet["payload"])
    altered["payload_sha256"] = hashlib.sha256(altered["payload"]).hexdigest()
    altered["handoff_digest"] = domain_digest(
        HANDOFF_SCHEMA, {k:v for k,v in altered.items() if k not in ("payload","handoff_digest")})
    with pytest.raises(ValueError, match="actual upstream"):
        admit_handoff(plan, altered, index=0, position=0, preceding=GENESIS,
                      approved_producers={"local-layer0"},
                      expected_vector=[0.125]*8)


def test_corrupt_transit_payload_rejected_before_next_layer():
    kwargs, *_ = setup()
    def corrupt(handoff):
        return dict(handoff, payload=handoff["payload"][:-8] + b"\x00"*8)
    with pytest.raises(ValueError, match="payload hash"):
        execute_pretrained_chain(**kwargs, transfer=corrupt)


def test_failed_late_handoff_rolls_back_prior_layer_caches(monkeypatch):
    kwargs, *_ = setup()
    from experiments.model_fabric import pretrained_chain as chain
    created = []
    original_init = chain.PreparedLayer.__init__
    def record(layer, *args, **kw):
        original_init(layer, *args, **kw)
        created.append(layer)
    monkeypatch.setattr(chain.PreparedLayer, "__init__", record)
    def fail(packet):
        raise ValueError("simulated transport failure")
    with pytest.raises(ValueError, match="simulated"):
        execute_pretrained_chain(**kwargs, transfer=fail)
    assert len(created) == 2
    assert all(layer.step == 0 and not layer.keys and not layer.values for layer in created)


def test_multi_layer_contract_refuses_unqualified_host_and_missing_route():
    kwargs, *_ = setup()
    bad = dict(kwargs)
    bad["approved_hosts"] = {-1: {"source-a"}, 0: {"source-a"}, 1: {"source-a"}}
    with pytest.raises(ValueError, match="Unqualified"):
        execute_pretrained_chain(**bad)
    bad = dict(kwargs)
    bad["range_hosts"] = {-1: kwargs["range_hosts"][-1], 0: kwargs["range_hosts"][0]}
    with pytest.raises(ValueError, match="Missing"):
        execute_pretrained_chain(**bad)
    bad = dict(kwargs, compute_actors={0: "unapproved"})
    with pytest.raises(ValueError, match="Unapproved"):
        execute_pretrained_chain(**bad)


def test_pretrained_tensor_manifest_compiles_layer1_without_weight_claim():
    from pathlib import Path
    import json
    path = Path(__file__).resolve().parents[1] / "experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json"
    if not path.exists():
        pytest.skip("repository manifest not installed in local reduced fixture")
    ground = json.loads(path.read_text())
    plan = compile_decoder_dag(ground)
    assert plan["architecture"]["layers"] == 30
    assert next(n for n in plan["nodes"] if n["id"] == "layer.1.input_norm")["inputs"] == ["layer.0.ffn_residual"]
    assert plan["weight_values_observed"] is False


def test_independent_two_layer_hidden_state_from_exact_captured_ranges():
    from collections import defaultdict
    from experiments.model_fabric.real_chain_lab import independent_two_layer_reference
    kwargs, _, _, _, _ = setup(dtype="BF16", tokens=[3,2,7])
    fragments = defaultdict(list)
    for i, host in kwargs["range_hosts"].items():
        def captured(request, host=host):
            reply = host(request)
            fragments[request["tensor"]].append((request["byte_begin"],reply["payload"]))
            return reply
        kwargs["range_hosts"][i] = captured
    result = execute_pretrained_chain(**kwargs)
    reference = independent_two_layer_reference(kwargs["plan"], fragments, kwargs["tokens"])
    assert len(reference) == len(result["output"])
    assert max(abs(x-y) for x,y in zip(reference,result["output"])) < 2e-6
