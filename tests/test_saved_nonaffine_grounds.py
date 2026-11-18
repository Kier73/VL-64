"""Versioned non-affine and observable quotient Grounds remain independently navigable."""
import json
from pathlib import Path

from vld import ObserverIntention
from experiments.byte_fabric.ground_families import GroundFamilies
from experiments.observer_intent.arbitrate import arbitrate, render_result


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/byte_fabric/datasets"


def stored(filename):
    return json.loads((DATA/filename).read_text(encoding="utf-8"))


def test_saved_nonaffine_and_quotient_navigate_without_original_frames():
    measurements=stored("2026-11-18-nonaffine-and-observable-measurements.json")
    raw=stored("2026-11-18-nonaffine-ground.json")
    observable=stored("2026-11-18-observable-quotient-ground.json")
    drivers=GroundFamilies()
    assert drivers.open(raw).realize(
        10**12,"digest.sha256")["result"]["value"]==measurements[
            "exact_discrete"]["trillion_result_digest"]
    assert drivers.open(observable).realize(
        10**12,"state.observable")["result"]["value"]==measurements[
            "observable_quotient"]["horizon_observed_value"]
    assert measurements["cross_runtime"]["discrete_ground_identity_equal"]
    assert measurements["cross_runtime"]["quotient_ground_identity_equal"]


def test_saved_ground_resolves_through_original_external_observer_interface():
    req=ObserverIntention(
        "ground.navigate",
        "workspace:experiments/byte_fabric/datasets/2026-11-18-nonaffine-ground.json",
        output="jsonl",
        options={"goal":{"kind":"digest.sha256","steps":10**12,
                         "max_work":4,"max_hops":4}})
    receipt=arbitrate(req,byte_root=ROOT)
    assert receipt["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert len(render_result(receipt,"jsonl").splitlines())==1
    req2=ObserverIntention(
        "ground.navigate",
        "workspace:experiments/byte_fabric/datasets/2026-11-18-observable-quotient-ground.json",
        output="text",
        options={"goal":{"kind":"state.observable","steps":10**12,
                         "max_work":2,"max_hops":2}})
    quotient_receipt=arbitrate(req2,byte_root=ROOT)
    assert quotient_receipt["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert "Result:" in render_result(quotient_receipt,"text")


def test_observable_ground_refuses_exact_byte_claim():
    q=stored("2026-11-18-observable-quotient-ground.json")
    assert q["reconstruction_claim"][
        "exact_source_bytes_recoverable_from_quotient_alone"] is False
    result=GroundFamilies().open(q).realize(1,"bytes.snapshot")
    assert result["status"]=="UNRESOLVED"
    assert result["reason"]=="QUOTIENT_CANNOT_REALIZE_EXACT_BYTES"


def test_measured_compression_is_scoped_to_exact_state_or_observable():
    m=stored("2026-11-18-nonaffine-and-observable-measurements.json")
    assert m["exact_discrete"]["compression_ratio"]>50
    assert m["exact_discrete"]["heldout_frames"]==52
    assert m["observable_quotient"]["source_states"]==12
    assert m["observable_quotient"]["quotient_states"]==3
    assert m["observable_quotient"]["exact_source_bytes_reconstructible"] is False
    assert m["cross_runtime"]["compact_core_bytes"]<=65536
