"""Versioned compressive Ground is independently navigable without snapshots."""
import json
from pathlib import Path

from vld import ObserverIntention
from experiments.byte_fabric.orbit_ground import ComposedGroundNavigator
from experiments.observer_intent.arbitrate import arbitrate,render_result


ROOT=Path(__file__).resolve().parents[1]
BASE="experiments/byte_fabric/datasets/"


def test_persisted_observed_ground_can_drive_fresh_observer():
    desc=json.loads((ROOT/(BASE+"2026-11-18-compressive-orbit-ground.json")).read_text())
    measurement=json.loads((ROOT/(BASE+"2026-11-18-compressive-orbit-measurements.json")).read_text())
    ground=ComposedGroundNavigator(desc)
    assert ground.check_against_core(10**12)
    req=ObserverIntention(
        "ground.navigate",
        "workspace:"+BASE+"2026-11-18-compressive-orbit-ground.json",
        output="jsonl",
        options={"goal":{"kind":"digest.sha256","steps":10**12,
                         "max_hops":4,"max_work":5}})
    outcome=arbitrate(req,byte_root=ROOT)
    assert outcome["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert len(render_result(outcome,"jsonl").splitlines())==1
    result=outcome["navigation"]
    assert result["result"]["value"]==measurement["constant_invariants"]["virtual_horizon_result_digest"]
    assert result["ground_digest"]==desc["ground_digest"]
    assert len(result["stage_receipts"])==4
    assert desc["interpreter"]==measurement["constant_invariants"]["discovered_interpreter"]
    assert measurement["cross_runtime_ground_agrees"] is True


def test_ground_description_is_not_merely_full_snapshot_archive():
    desc=json.loads((ROOT/(BASE+"2026-11-18-compressive-orbit-ground.json")).read_text())
    measurement=json.loads((ROOT/(BASE+"2026-11-18-compressive-orbit-measurements.json")).read_text())
    stored=len(json.dumps(desc,sort_keys=True,separators=(",",":")).encode())
    assert stored==measurement["constant_invariants"]["ground_description_bytes"]
    assert stored < 2000
    assert measurement["constant_invariants"]["source_snapshot_bytes"]==81920
    assert "frames" not in desc
    assert "future_states_proven" not in desc
    assert desc["scope"]=="single-fixed-template-affine-observed-hypothesis"
