"""MPF-002: actual host-resident trained numerical behavior, pinned source sessions.

The byte LM is a real trained operator fixture, not a claim of running the
public DeepSeek/Qwen weights.  The bridge accepts source-local remote callables
only if their observed identity matches independent caller pins.
"""
import copy
import json
import subprocess
import sys

import pytest

from experiments.model_fabric.byte_lm import fit_byte_lm
from experiments.model_fabric.operational_ground import GROUND_SCHEMA, digest
from experiments.model_fabric.operational_hosts import HostedTrainedByteLanguage
from experiments.model_fabric.source_pinned_behavior_fleet import SourcePinnedBehaviorFleet
from experiments.model_fabric.virtual_page_fleet import (
    HostedModelSource, VirtualModelPageFleet, ModelPageRefused, _hash,
)

TRAINING = (b"Observer grounded computation. An operator navigates trained "
            b"continuations. Independent observers preserve their causal "
            b"identity across the same resident model. ") * 16


def source(name="lab/resident-trained-model", revision="a" * 40):
    record = {
        "schema": "mailcpu.external-virtual-structure.v1.source",
        "repository": name, "revision": revision,
        "files": [{"path": "config.json", "size_bytes": 180, "source_sha256": None}],
        "config": {"model_type": "count_trained_byte", "num_hidden_layers": 1},
        "observation_digests": ["b" * 64],
        "scope": "external_structure_only_zero_model_payload",
    }
    return HostedModelSource(record, expected_head=_hash(record))


def resident(src, *, host_id="numerical-host"):
    learned = fit_byte_lm(TRAINING)
    host = HostedTrainedByteLanguage(learned, observer_host=host_id)
    # Test adapter: associate the host-owned trained fixture with a synthetic
    # source identity; this is NOT a public checkpoint authenticity assertion.
    host.description["model"] = src.repository
    host.description["revision"] = src.revision
    host.bound_ground = digest(GROUND_SCHEMA, host.description)
    return host, learned["ground_digest"]


def make_fleet(n=4, *, alias="trained", src=None, host=None, trained_sha=None):
    src = src or source()
    host, trained_sha = (host, trained_sha) if host else resident(src)
    structure = VirtualModelPageFleet()
    structure.register(alias, src)
    result = SourcePinnedBehaviorFleet(structure)
    result.bind(alias, host, approved_hosts={host.description["observer_host"]},
                expected_artifact=trained_sha)
    for i in range(n):
        structure.open(f"reader-{i}", alias, host_id="receiver-A")
        result.open(f"reader-{i}")
    return result, host, trained_sha


def test_one_trained_host_many_concurrent_causal_sessions_and_zero_weights():
    fleet, host, artifact = make_fleet(12)
    requests = [
        {"observer_id": f"reader-{i}", "user_text": f"Compute sample {i}.",
         "request_id": "turn-1", "max_new_tokens": 4, "tile_size": 41}
        for i in range(12)
    ]
    receipts = fleet.turn_many(requests, workers=12)
    assert len(receipts) == 12
    assert len({r["head"] for r in receipts}) == 12
    assert len({r["source_head"] for r in receipts}) == 1
    assert len({r["operator_ground"] for r in receipts}) == 1
    assert all(len(r["observation"]["generated_tokens"]) == 4 for r in receipts)
    assert all(r["observation"]["host_calls"] >= 4*7 for r in receipts)
    assert all(r["model_payload_bytes"] == 0 for r in receipts)
    assert all(r["observation"]["remote_weight_bytes_received"] == 0 for r in receipts)
    assert all(r["checkpoint"]["model_checkpoint"]["tokens"] for r in receipts)
    snap = fleet.snapshot()
    assert "sparse_weights" not in json.dumps(snap)
    assert "training_sha256" not in json.dumps(snap)
    assert "model.safetensors" not in json.dumps(snap)
    restored = SourcePinnedBehaviorFleet.restore(
        snap, expected_digest=snap["checkpoint_digest"])
    assert restored.snapshot() == snap
    assert len(host.sessions) <= 8  # existing resident-host bounded cache
    assert artifact == receipts[0]["checkpoint"]["model_checkpoint"].get("model_artifact_identity", artifact)


def test_two_actual_dialogue_turns_preserve_context_and_idempotency():
    fleet, host, artifact = make_fleet(1)
    original = fleet.state("reader-0")["tip"]
    first = fleet.turn("reader-0", "A grounded starting question.",
                       request_id="first", max_new_tokens=3, expected_tip=original)
    before = fleet.state("reader-0")
    calls = len(host.sessions)
    assert fleet.turn("reader-0", "A grounded starting question.",
                      request_id="first", max_new_tokens=3, expected_tip=original) == first
    assert len(host.sessions) == calls
    with pytest.raises(ModelPageRefused, match="reused"):
        fleet.turn("reader-0", "A different question.", request_id="first")
    with pytest.raises(ModelPageRefused, match="stale"):
        fleet.turn("reader-0", "Next question.", request_id="next", expected_tip=original)
    second = fleet.turn("reader-0", "A second question.", request_id="second",
                        max_new_tokens=3, expected_tip=before["tip"])
    assert second["sequence"] == 1
    assert second["predecessor"] == first["head"]
    assert second["model_intention"]["messages"][-2]["role"] == "assistant"
    assert second["model_intention"]["messages"][-1]["role"] == "user"
    assert first["observation"]["transcript_head"] != second["observation"]["transcript_head"]
    assert fleet.state("reader-0")["sequence"] == 2


def test_source_rev_artifact_host_and_observer_substitution_refused():
    src = source()
    host, artifact = resident(src)
    f = VirtualModelPageFleet()
    f.register("one", src)
    f.register("two", source("lab/different-model", "b" * 40))
    f.open("a", "one")
    f.open("b", "two")
    b = SourcePinnedBehaviorFleet(f)
    with pytest.raises(ValueError, match="artifact"):
        b.bind("one", host, approved_hosts={"numerical-host"},
               expected_artifact="c" * 64)
    with pytest.raises(ValueError, match="Host model"):
        b.bind("two", host, approved_hosts={"numerical-host"},
               expected_artifact=artifact)
    with pytest.raises(ValueError, match="Host model"):
        b.bind("one", host, approved_hosts={"wrong-host"},
               expected_artifact=artifact)
    b.bind("one", host, approved_hosts={"numerical-host"}, expected_artifact=artifact)
    with pytest.raises(ModelPageRefused):
        b.open("b")
    b.open("a")
    with pytest.raises(ModelPageRefused):
        b.turn("b", "Not my source.", request_id="wrong")
    with pytest.raises(ValueError):
        b.turn("a", "Question.", request_id="invalid", max_new_tokens=0)


def test_source_observer_handoff_fences_stale_behavior_and_reanchors():
    fleet, host, artifact = make_fleet(1)
    first = fleet.turn("reader-0", "Before host handoff.", request_id="first",
                       max_new_tokens=2)
    old = fleet.state("reader-0")["tip"]
    s = fleet.sources.state("reader-0")
    fleet.sources.rehome("reader-0", "receiver-B", expected_tip=s["tip"])
    with pytest.raises(ModelPageRefused, match="reanchor"):
        fleet.turn("reader-0", "Fenced.", request_id="stale", max_new_tokens=2)
    ev = fleet.reanchor("reader-0", expected_tip=old)
    assert ev["new_epoch"] == 1 and ev["predecessor"] == first["head"]
    after = fleet.turn("reader-0", "Continue on new receiver.", request_id="after",
                       max_new_tokens=2, expected_tip=ev["head"])
    assert after["epoch"] == 1 and after["predecessor"] == ev["head"]
    snap = fleet.snapshot()
    successor = SourcePinnedBehaviorFleet.restore(
        snap, expected_digest=snap["checkpoint_digest"])
    with pytest.raises(ModelPageRefused, match="reattached"):
        successor.turn("reader-0", "Cannot infer until host bound.",
                       request_id="missing", max_new_tokens=2)
    successor.bind("trained", host, approved_hosts={"numerical-host"},
                   expected_artifact=artifact)
    result = successor.turn("reader-0", "After independent recovery.",
                            request_id="recovered", max_new_tokens=2)
    assert result["epoch"] == 1
    assert result["source_head"] == first["source_head"]


def test_checkpoint_causal_corruption_foreign_source_and_intention_tampering_rejected():
    fleet, _, _ = make_fleet(2)
    fleet.turn("reader-0", "Alpha.", request_id="one", max_new_tokens=2)
    fleet.turn("reader-1", "Beta.", request_id="two", max_new_tokens=2)
    snap = fleet.snapshot()
    for change in (
        lambda d: d["events"][0].update(observer="reader-1"),
        lambda d: d["events"][0]["observation"].update(remote_weight_bytes_received=15),
        lambda d: d["events"][0]["checkpoint"]["model_checkpoint"].update(tokens=[2]),
        lambda d: d["operators"]["trained"].update(revision="c" * 40),
    ):
        altered = copy.deepcopy(snap)
        change(altered)
        with pytest.raises(ModelPageRefused):
            SourcePinnedBehaviorFleet.restore(
                altered, expected_digest=snap["checkpoint_digest"])


def test_reconstruction_in_separate_python_process_without_trained_host():
    fleet, _, _ = make_fleet(2)
    fleet.turn_many([
        {"observer_id": "reader-0", "user_text": "Observe A.",
         "request_id": "A", "max_new_tokens": 2},
        {"observer_id": "reader-1", "user_text": "Observe B.",
         "request_id": "B", "max_new_tokens": 2},
    ])
    snap = fleet.snapshot()
    code = ("import json,sys;"
            "from experiments.model_fabric.source_pinned_behavior_fleet import SourcePinnedBehaviorFleet;"
            "s=json.load(sys.stdin);"
            "r=SourcePinnedBehaviorFleet.restore(s,expected_digest=s['checkpoint_digest']);"
            "print(json.dumps({'digest':r.snapshot()['checkpoint_digest'],"
            "'sessions':len(r._sessions),'events':len(r._events)}))")
    out = subprocess.run([sys.executable, "-c", code], input=json.dumps(snap),
                         text=True, capture_output=True, check=True, timeout=30)
    observed = json.loads(out.stdout)
    assert observed == {"digest": snap["checkpoint_digest"], "sessions": 2, "events": 2}


def test_two_different_host_models_concurrent_and_source_isolated():
    a, b = source("lab/trained-A", "a"*40), source("lab/trained-B", "b"*40)
    h1, sha = resident(a, host_id="host-A")
    h2, sha2 = resident(b, host_id="host-B")
    f = VirtualModelPageFleet()
    f.register("A", a)
    f.register("B", b)
    f.open("observer-A", "A")
    f.open("observer-B", "B")
    bridge = SourcePinnedBehaviorFleet(f)
    bridge.bind("A", h1, approved_hosts={"host-A"}, expected_artifact=sha)
    bridge.bind("B", h2, approved_hosts={"host-B"}, expected_artifact=sha2)
    bridge.open("observer-A")
    bridge.open("observer-B")
    a_out, b_out = bridge.turn_many([
        {"observer_id": "observer-A", "user_text": "Same task.",
         "request_id": "task", "max_new_tokens": 2},
        {"observer_id": "observer-B", "user_text": "Same task.",
         "request_id": "task", "max_new_tokens": 2},
    ], workers=2)
    assert a_out["source_head"] != b_out["source_head"]
    assert a_out["operator_ground"] != b_out["operator_ground"]
    assert a_out["head"] != b_out["head"]
    recovered = SourcePinnedBehaviorFleet.restore(
        bridge.snapshot(), expected_digest=bridge.snapshot()["checkpoint_digest"])
    assert recovered.state("observer-A")["model"] == "A"
    assert recovered.state("observer-B")["model"] == "B"


def test_replacement_claiming_same_artifact_but_wrong_logits_is_refused():
    fleet, original, artifact = make_fleet(1)
    fleet.turn("reader-0", "Commit a causal numerical prefix.",
               request_id="one", max_new_tokens=2)
    snap = fleet.snapshot()
    restored = SourcePinnedBehaviorFleet.restore(
        snap, expected_digest=snap["checkpoint_digest"])
    wrong, _ = resident(fleet.sources._models["trained"])
    wrong.description = copy.deepcopy(original.description)
    wrong.bound_ground = digest(GROUND_SCHEMA, wrong.description)
    wrong.calculate_logits = lambda tokens: [0.0]*256
    with pytest.raises(ModelPageRefused, match="causal-logits"):
        restored.bind("trained", wrong,
                      approved_hosts={"numerical-host"},
                      expected_artifact=artifact)
    with pytest.raises(ModelPageRefused, match="reattached"):
        restored.turn("reader-0", "No replacement was admitted.",
                      request_id="not-admitted", max_new_tokens=2)
    restored.bind("trained", original,
                  approved_hosts={"numerical-host"},
                  expected_artifact=artifact)
    resumed = restored.turn("reader-0", "Correct host continues.",
                            request_id="continued", max_new_tokens=2)
    assert resumed["predecessor"] == snap["sessions"]["reader-0"]["tip"]
