"""Observer can send known byte intervals, not full raw frame."""
import base64
import json

from vld import ObserverIntention
from experiments.byte_fabric.multifield_lab import (
    experiment,source_template,frame,observed_edges,
)
from experiments.byte_fabric.multifield_ground import discover_multifield_ground
from experiments.observer_intent.arbitrate import arbitrate,render_result


def save_ground(tmp_path,ground):
    (tmp_path/"ground.json").write_text(json.dumps(ground))


def test_external_observer_navigation_uses_three_state_bytes_not_512_frame(tmp_path):
    sample=experiment()
    save_ground(tmp_path,sample["ground"])
    req=ObserverIntention(
        "ground.multifield.navigate","workspace:ground.json",output="json",
        options={"goal":{
            "initial_state_hex":"0000ff",
            "action":"tick","steps":10**12+3,
            "kind":"digest.sha256",
        }})
    outcome=arbitrate(req,byte_root=tmp_path)
    assert outcome["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert outcome["result"]["result"]["value"]==sample[
        "report"]["modeled_result_sha256"]
    assert outcome["result"]["cycle_skipped_steps"]>10**12-20
    assert outcome["plan"]["authority"]==(
        "no_external_actions_or_unobserved_byte_claims")
    assert json.loads(render_result(outcome,"json"))["status"]==outcome["status"]
    assert len(render_result(outcome,"jsonl").splitlines())==1


def test_external_observer_can_send_single_byte_and_receive_suggested_probe(tmp_path):
    sample=experiment()
    save_ground(tmp_path,sample["ground"])
    req=ObserverIntention(
        "ground.multifield.observe","workspace:ground.json",output="json",
        options={"goal":{
            "known_intervals":[[73,74]],
            "segments_b64":[base64.b64encode(b"\x01").decode("ascii")],
        }})
    outcome=arbitrate(req,byte_root=tmp_path)
    assert outcome["status"]=="MULTIPLE_COMPATIBLE_MODEL_STATES"
    state=outcome["result"]
    assert state["known_byte_count"]==1
    assert len(state["compatible_model_states"])==4
    assert state["next_probe"]["byte_interval"]==[311,313]
    assert state["next_probe"]["expected_information_bits"]==2.0
    assert state["scope"]=="observed_positions_only_unobserved_bytes_unverified"
    assert "Next observation:" in render_result(outcome,"text")


def test_external_observer_refuses_unqualified_partial_claims(tmp_path):
    sample=experiment()
    save_ground(tmp_path,sample["ground"])
    for intervals,parts in (
        ([[311,313]],["AQ=="]),  # one byte, claimed two
        ([[72,75],[73,74]],["AAAA","AQ=="]),  # overlap
        ([[1000,1001]],["AQ=="]),  # outside frame
    ):
        req=ObserverIntention(
            "ground.multifield.observe","workspace:ground.json",
            options={"goal":{"known_intervals":intervals,"segments_b64":parts}})
        assert arbitrate(req,byte_root=tmp_path)["status"]=="UNRESOLVED"
    unknown=ObserverIntention(
        "ground.multifield.navigate","workspace:ground.json",
        options={"goal":{
            "initial_state_hex":"0000ff",
            "action":"shell","steps":1,"kind":"digest.sha256"}})
    assert arbitrate(unknown,byte_root=tmp_path)["status"]=="UNRESOLVED"


def test_external_observer_refuses_missing_state_action_without_fabricating(tmp_path):
    base=source_template()
    frames=observed_edges(base,omit=((1,3,"tick"),))
    partial=discover_multifield_ground(frames)
    save_ground(tmp_path,partial["ground"])
    req=ObserverIntention(
        "ground.multifield.navigate","workspace:ground.json",
        options={"goal":{
            "initial_state_hex":"0103fc",
            "action":"tick","steps":10**12,
            "kind":"bytes.snapshot"}})
    outcome=arbitrate(req,byte_root=tmp_path)
    assert outcome["status"]=="UNRESOLVED"
    assert outcome["result"]["reason"]=="UNOBSERVED_STATE_ACTION"
    assert outcome["result"]["completed_steps"]==0



def test_external_observer_composes_mixed_action_program_from_typed_intention(tmp_path):
    sample=experiment()
    save_ground(tmp_path,sample["ground"])
    word=["tick","toggle","tick","tick","toggle"]
    req=ObserverIntention(
        "ground.multifield.navigate","workspace:ground.json",output="json",
        options={"goal":{"initial_state_hex":"0000ff",
                         "actions":word,"kind":"state.token"}})
    response=arbitrate(req,byte_root=tmp_path)
    assert response["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert response["result"]["completed_steps"]==5
    assert [edge["action"] for edge in response[
        "result"]["witnessed_edge_receipts"]]==word
    assert response["plan"]["authority"]==(
        "no_external_actions_or_unobserved_byte_claims")
