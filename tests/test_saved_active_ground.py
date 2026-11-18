"""Saved action Ground and recorded active experiments are reusable in fresh runs."""
from __future__ import annotations

import json
from pathlib import Path

from vld import ObserverIntention,domain_digest
from experiments.byte_fabric.active_ground import (
    ActionGroundNavigator,ActiveGroundSession,
)
from experiments.observer_intent.arbitrate import arbitrate,render_result


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/byte_fabric/datasets"


def stored(filename):
    return json.loads((DATA/filename).read_text(encoding="utf-8"))


def test_saved_model_and_action_probes_reproduce_the_selection():
    catalogue=stored("2026-11-18-active-candidates.json")
    model=stored("2026-11-18-active-selected-ground.json")
    session=stored("2026-11-18-active-session.json")
    evidence=stored("2026-11-18-active-measurements.json")
    planned=ActiveGroundSession(catalogue["candidates"],
                                costs=catalogue["costs"]).plan(
        max_depth=2,max_words=32)
    assert planned["word"]==["advance","inspect"]
    assert planned["expected_information_bits"]==1.0
    assert evidence["controlled"]["action_words"]==[
        ["advance","inspect"],["switch","inspect"]]
    assert session["status"]=="ONE_HYPOTHESIS_REMAINS"
    assert session["remaining_ground_digests"]==[model["ground_digest"]]
    assert evidence["cross_runtime"]["session_digest_equal"] is True
    assert session["session_digest"]==domain_digest(
        "vl64.active-ground-session.v1",
        {k:v for k,v in session.items() if k!="session_digest"})
    assert ActionGroundNavigator(model).repeat("advance",10**12)[
        "terminal_state"]==evidence["controlled"]["modeled_terminal_state"]


def test_saved_ground_intentions_are_pure_observer_communication():
    for filename,expected in [
        ("active_next_probe.json","QUALIFIED_INFORMATIVE_PROBE"),
        ("active_action_trillion.json","MODELED_ACTION_NAVIGATION"),
    ]:
        name="experiments/byte_fabric/intents/"+filename
        req=ObserverIntention.from_dict(json.loads((ROOT/name).read_text()))
        observed=arbitrate(req,byte_root=ROOT)
        assert observed["status"]==expected
        assert observed["plan"]["execution_authority"]==(
            "no_host_actions_performed_by_this_observer_route")
        assert json.loads(render_result(observed,"json"))["status"]==expected


def test_recorded_web_action_is_real_readonly_bytes_not_remote_state_claim():
    record=stored("2026-11-18-active-measurements.json")
    web=record["real_readonly_web"]
    assert web["status"]=="ONE_HYPOTHESIS_REMAINS"
    assert web["selected_actions"]==["read_readme"]
    assert web["host_policy"]=="read-only-remote"
    assert all(x["capture_status"]=="COMPLETE" and x["observed_bytes"]>1000
               for x in web["initial_byte_observations"])
    assert web["scope"]==(
        "read-only-resource-identity-discrimination-not-remote-stateful-automata-learning")
    assert record["cross_runtime"]["remote_bytes_sha256_equal"]
    assert record["cross_runtime"]["compact_core_source_bytes"]<=65536
    assert record["limitation"].startswith("Finite candidate elimination")


def test_distinct_complete_action_model_cannot_be_made_by_truncating_a_saved_one():
    from experiments.byte_fabric.active_ground import ActionHypothesis
    selected=stored("2026-11-18-active-selected-ground.json")
    changed=dict(selected)
    changed["states"]=[dict(x) for x in selected["states"]]
    changed["states"][0]={k:v for k,v in changed["states"][0].items()
                         if k!="switch"}
    changed["ground_digest"]=domain_digest(
        "vl64.action-conditioned-ground.v1",
        {k:v for k,v in changed.items() if k!="ground_digest"})
    import pytest
    with pytest.raises(ValueError,match="Incomplete"):
        ActionHypothesis(changed)
