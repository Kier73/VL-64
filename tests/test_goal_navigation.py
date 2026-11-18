"""Observer asks for results, not a hardcoded operation or architecture."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path

import pytest
from vld import ObserverIntention
from experiments.byte_fabric.ground import (
    QualifiedByteExecutor, OperationCapability, ground_for_capture,
)
from experiments.byte_fabric.navigator import GoalNavigator, GoalRelation, validate_goal
from experiments.byte_fabric.sources import capture_bytes, acquire
from experiments.byte_fabric.extensions import (
    SourceRegistry, SourceCapability, DecoderRegistry,
)
from experiments.byte_fabric.router import (
    byte_arbitrate, compile_byte_plan, replay_byte_receipt,
)
from experiments.observer_intent.arbitrate import arbitrate, render_result

ROOT = Path(__file__).resolve().parents[1]
JSON_TARGET = "workspace:experiments/byte_fabric/fixtures/state.json"
HTML_TARGET = "workspace:experiments/byte_fabric/fixtures/page.html"


def ask(target, kind, *, parameters=None, expected=None, max_work=100, max_bytes=65536):
    goal = {"kind": kind, "parameters": parameters or {}, "max_work": max_work}
    if expected is not None:
        goal["expected_result"] = expected
    return ObserverIntention("bytes.realize", target, output="json", options={
        "goal": goal, "max_bytes": max_bytes})


def test_observer_requests_json_value_not_json_pointer_operator():
    req = ask(JSON_TARGET, "value.json", parameters={
        "path": "/observer/constraints/verified"})
    plan = compile_byte_plan(req)
    assert plan["kind"] == "GOAL_NAVIGATION"
    assert plan["steps"][-2:] == ["realize_or_refuse", "bind_result_receipt"]
    receipt = arbitrate(req, byte_root=ROOT)
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert receipt["execution"]["result"] == {"value": True, "value_type": "bool"}
    navigation = receipt["goal_navigation"]
    assert navigation["selected"]["relation"] == "json-field"
    assert navigation["selected"]["operator"] == "json.pointer"
    assert navigation["navigation"]["status"] == "QUALIFIED_ROUTE_AVAILABLE"
    assert navigation["attempted_relations"][0]["status"] == "EXECUTED_LOCAL"
    captured = acquire(req.spec["target"], root=ROOT)
    check = replay_byte_receipt(receipt, capture=captured)
    assert check["status"] == "VERIFIED_BYTE_GROUND"
    assert check["operations_replayed"] == 1
    assert json.loads(render_result(receipt, "json"))["status"] == "EXECUTED_LOCAL"
    assert len(render_result(receipt, "jsonl").splitlines()) == 1
    assert "Goal navigation:" in render_result(receipt, "text")


def test_arbitrary_binary_digest_by_goal_and_exact_replay():
    raw = bytes(range(0, 126)) + b"\xff\x00"
    target = "inline:b64," + base64.b64encode(raw).decode()
    req = ask(target, "digest.sha256")
    receipt = byte_arbitrate(req)
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert receipt["ground"]["view"]["kind"] == "opaque-bytes"
    assert receipt["execution"]["result"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["goal_navigation"]["selected"]["relation"] == "digest-from-bytes"
    capture = acquire(target, root=ROOT)
    assert replay_byte_receipt(receipt,capture=capture)["status"] == "VERIFIED_BYTE_GROUND"
    receipt["goal_navigation"]["selected"]["relation"] = "forged"
    assert replay_byte_receipt(receipt,capture=capture)["reason"] == "GOAL_REALIZATION_MISMATCH"


def test_html_links_goal_and_byte_slice_goal():
    req = ask(HTML_TARGET, "links.html")
    receipt = arbitrate(req, byte_root=ROOT)
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert "/docs/USAGE_GUIDE.md" in receipt["execution"]["result"]["links"]
    target = "inline:b64," + base64.b64encode(b"\x00\xffhello").decode()
    request = ask(target, "bytes.interval", parameters={"start": 1, "length": 4})
    result = arbitrate(request)
    assert result["status"] == "EXECUTED_LOCAL"
    assert base64.b64decode(result["execution"]["result"]["data"]) == b"\xffhel"


def test_no_qualified_relation_does_not_acquire_source(monkeypatch):
    import experiments.byte_fabric.router as routing
    def no_network(*args, **kwargs):
        pytest.fail("Source must not be acquired for an unadmitted goal")
    monkeypatch.setattr(routing, "acquire", no_network)
    req = ask("https://example.com/", "model.infer")
    assert compile_byte_plan(req)["status"] == "UNRESOLVED"
    assert byte_arbitrate(req)["reason"] == "NO_QUALIFIED_GOAL_RELATION"
    assert byte_arbitrate(ask("https://example.com/", "digest.sha256",
                             max_work=0))["reason"] == "INVALID_OBSERVER_GOAL"
    invalid = ObserverIntention("bytes.realize", "https://example.com/",
                    options={"goal":{"kind":"value.json", "parameters": []}})
    assert byte_arbitrate(invalid)["status"] == "REFUSED"


def test_input_view_guard_and_incomplete_source_are_enforced(tmp_path):
    binary = b"\xff" * 200
    (tmp_path / "opaque.bin").write_bytes(binary)
    target = "workspace:opaque.bin"
    typed = byte_arbitrate(ask(target, "value.json",
                         parameters={"path":"/id"}),root=tmp_path)
    assert typed["status"] == "UNRESOLVED"
    assert typed["goal_navigation"]["navigation"]["candidate_relations"][0]["reason"] == "REQUIRED_VIEW_UNOBSERVED"
    partial = byte_arbitrate(ask(target, "digest.sha256",max_bytes=12),root=tmp_path)
    assert partial["status"] == "UNRESOLVED"
    assert partial["ground"]["observation"]["capture_status"] == "PREFIX"
    assert partial["goal_navigation"]["navigation"]["candidate_relations"][0]["reason"] == "COMPLETE_SOURCE_REQUIRED"
    sampled = byte_arbitrate(ask(target, "bytes.interval",
                           parameters={"start":3, "length":4},
                           max_bytes=12),root=tmp_path)
    assert sampled["status"] == "EXECUTED_LOCAL"
    assert sampled["execution"]["input_scope"] == "prefix"


def test_work_budget_and_expected_result_are_acceptance_conditions():
    req = ask(JSON_TARGET, "stats.bytes", max_work=2)
    result = arbitrate(req,byte_root=ROOT)
    assert result["status"] == "UNRESOLVED"
    assert result["goal_navigation"]["navigation"]["candidate_relations"][0]["reason"] == "COST_CEILING"
    accepted = arbitrate(ask(JSON_TARGET, "stats.bytes", max_work=3),byte_root=ROOT)
    assert accepted["status"] == "EXECUTED_LOCAL"
    wrong = arbitrate(ask(JSON_TARGET, "value.json",
                        parameters={"path":"/observer/intent"},
                        expected={"value":"NOT_THE_VALUE","value_type":"str"}),byte_root=ROOT)
    assert wrong["status"] == "UNRESOLVED"
    assert wrong["goal_navigation"]["attempted_relations"][0]["status"] == "REJECTED_EXPECTED_RESULT"
    assert wrong["goal_navigation"]["execution"] is None


def test_custom_sensor_decoder_and_host_realizer_are_general():
    sources = SourceRegistry()
    sources.register(SourceCapability("sensor:"),
        lambda target,n: capture_bytes(struct.pack("<I",73), target))
    decoders = DecoderRegistry()
    decoders.register("register32",
        lambda payload,hint: len(payload)==4,
        lambda payload,hint: {"register_value":int.from_bytes(payload,"little")})
    executor = QualifiedByteExecutor()
    executor.register(OperationCapability("read.sensor"),
        lambda payload, params: {"reading":int.from_bytes(payload,"little")})
    navigator = GoalNavigator(executor=executor)
    navigator.register(GoalRelation(
        "read-sensor-u32", "value.sensor", "read.sensor",
        requires_view="register32",estimated_work=4))
    req = ask("sensor:temperature", "value.sensor",max_work=4)
    assert compile_byte_plan(req,sources=sources,executor=executor,
                             navigator=navigator)["kind"] == "GOAL_NAVIGATION"
    result = byte_arbitrate(req,sources=sources,decoders=decoders,
                          executor=executor,navigator=navigator)
    assert result["status"] == "EXECUTED_LOCAL"
    assert result["execution"]["result"]["reading"] == 73
    assert result["goal_navigation"]["selected"]["relation"] == "read-sensor-u32"
    cap = sources.acquire("sensor:temperature",max_bytes=65536)
    assert replay_byte_receipt(result,capture=cap,decoders=decoders,
            executor=executor,navigator=navigator)["status"] == "VERIFIED_BYTE_GROUND"


def test_choose_least_estimated_work_not_first_registered():
    executor = QualifiedByteExecutor()
    executor.register(OperationCapability("slow.identity"),
        lambda payload, params: {"length":len(payload)})
    executor.register(OperationCapability("fast.identity"),
        lambda payload, params: {"length":len(payload)})
    nav = GoalNavigator(executor=executor, relations=[])
    nav.register(GoalRelation("slow-method","value.length","slow.identity",
                              estimated_work=20))
    nav.register(GoalRelation("fast-method","value.length","fast.identity",
                              estimated_work=2))
    req = ask(JSON_TARGET, "value.length")
    receipt = byte_arbitrate(req,root=ROOT,executor=executor,navigator=nav)
    assert receipt["status"] == "EXECUTED_LOCAL"
    assert receipt["goal_navigation"]["selected"]["relation"] == "fast-method"
    assert len(receipt["goal_navigation"]["navigation"]["candidate_relations"]) == 2
    assert len(receipt["goal_navigation"]["attempted_relations"]) == 1
    assert receipt["execution"]["result"]["length"] > 0


def test_cannot_register_unknown_or_side_effecting_realizer():
    navigator = GoalNavigator()
    with pytest.raises(ValueError):
        navigator.register(GoalRelation("arbitrary","value.remote","shell.exec"))
    with pytest.raises(ValueError):
        navigator.register(GoalRelation("bad", "value.remote","sha256",
                                        estimated_work=-1))
    with pytest.raises(ValueError):
        navigator.register(GoalRelation("bad", "value.remote","sha256",
                                        parameters=(("oops",),)))


def test_same_goal_digest_across_hosts_with_distinct_capture_receipts():
    data = b"common computational state"
    req = ask("inline:b64," + base64.b64encode(data).decode(),"digest.sha256")
    captured = acquire(req.spec["target"],root=ROOT)
    ground_a = ground_for_capture(capture_bytes(data,"host:A",
                transport={"ms":8}))
    ground_b = ground_for_capture(capture_bytes(data,"host:B",
                transport={"ms":300}))
    assert ground_a["ground_digest"] == ground_b["ground_digest"]
    nav = GoalNavigator()
    a = nav.realize(req.spec["options"]["goal"],ground_a,capture_bytes(
        data,"host:A",transport={"ms":8}))
    b = nav.realize(req.spec["options"]["goal"],ground_b,capture_bytes(
        data,"host:B",transport={"ms":300}))
    assert a["execution"]["result"] == b["execution"]["result"]
    assert a["navigation"]["plan_digest"] == b["navigation"]["plan_digest"]
