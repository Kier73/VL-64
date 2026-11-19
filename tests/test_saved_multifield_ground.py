"""Stored multi-field Ground and sparse observations are independently usable."""
import json
from pathlib import Path

from vld import ObserverIntention
from experiments.byte_fabric.multifield_ground import MultiFieldNavigator
from experiments.observer_intent.arbitrate import arbitrate,render_result


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/byte_fabric/datasets"


def stored(filename):
    return json.loads((DATA/filename).read_text())


def test_saved_multifield_ground_navigates_without_original_training_frames():
    ground=stored("2026-11-18-multifield-ground.json")
    measured=stored("2026-11-18-multifield-measurements.json")
    nav=MultiFieldNavigator(ground)
    assert measured["cross_runtime_ground_equal"]
    assert ground["ground_digest"]==measured["ground_digest"]
    assert nav.navigate("0000ff","tick",10**12+3,"digest.sha256")[
        "result"]["value"]==measured["measurements"][0][
            "modeled_result_sha256"]
    assert nav.navigate("0000ff","tick",10**12+3)[
        "evaluated_transitions"]<=8


def test_saved_observer_sparse_query_has_only_one_known_byte():
    req=ObserverIntention.from_dict(json.loads((
        ROOT/"experiments/byte_fabric/intents/multifield_recommend_field.json"
    ).read_text()))
    result=arbitrate(req,byte_root=ROOT)
    assert result["status"]=="MULTIPLE_COMPATIBLE_MODEL_STATES"
    assert result["result"]["known_byte_count"]==1
    assert result["result"]["next_probe"]["byte_interval"]==[311,313]
    assert result["result"]["next_probe"]["expected_information_bits"]==2
    assert json.loads(render_result(result,"json"))["status"]==result["status"]


def test_saved_trillion_and_mixed_program_intentions():
    root=ROOT/"experiments/byte_fabric/intents"
    for filename in ("multifield_trillion.json","multifield_mixed_program.json"):
        req=ObserverIntention.from_dict(json.loads((root/filename).read_text()))
        out=arbitrate(req,byte_root=ROOT)
        assert out["status"]=="OBSERVED_GROUND_NAVIGATION"
        assert out["plan"]["authority"]=="no_external_actions_or_unobserved_byte_claims"
        assert out["result"]["ground_digest"]==stored(
            "2026-11-18-multifield-ground.json")["ground_digest"]
    mixed_req=ObserverIntention.from_dict(json.loads((
        root/"multifield_mixed_program.json").read_text()))
    mixed=arbitrate(mixed_req,byte_root=ROOT)
    assert mixed["result"]["completed_steps"]==5
    assert len(mixed["result"]["witnessed_edge_receipts"])==5


def test_saved_multifield_measurements_preserve_scope_and_unknowns():
    m=stored("2026-11-18-multifield-measurements.json")
    for observation in m["measurements"]:
        assert observation["known_transition_edges"]==16
        assert observation["unseen_mixed_action_words_checked"]==62
        assert observation["missing_edge_status"]=="UNRESOLVED"
        assert observation["tampered_byte_status"]=="CONTRADICTED"
        assert observation["witnessed_cross_field_dependency"]
        assert observation["compression_ratio"]>3
    assert "not general arbitrary architecture inference" in m["scope"]
