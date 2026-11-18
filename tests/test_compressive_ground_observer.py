"""Observer contract for a compressed Ground, independent of source format."""
import base64
import json
from pathlib import Path

from vld import ObserverIntention
from experiments.byte_fabric.orbit_lab import experiment
from experiments.byte_fabric.orbit_adapter import arbitrate_orbit,render_orbit_receipt


def test_observer_navigates_saved_ground_in_requested_format(tmp_path):
    results=experiment()
    descriptor=tmp_path/"ground.json"
    descriptor.write_text(json.dumps(results["ground"]))
    req=ObserverIntention(
        "ground.navigate","workspace:ground.json",output="json",
        options={"goal":{"steps":10**12,"kind":"digest.sha256",
                         "max_hops":4,"max_work":5}})
    realized=arbitrate_orbit(req,root=tmp_path)
    assert realized["status"]=="OBSERVED_GROUND_NAVIGATION"
    nav=realized["navigation"]
    assert nav["result"]["value"] == results["report"]["horizon_output_digest"]
    assert len(nav["stage_receipts"]) == 4
    assert [x["relation"] for x in nav["stage_receipts"]] == [
        "interpret-span","navigate-affine","materialize-span",
        "digest-realization"]
    assert json.loads(render_orbit_receipt(realized,"json"))["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert len(render_orbit_receipt(realized,"jsonl").splitlines())==1
    assert "Horizon: 1000000000000" in render_orbit_receipt(realized,"text")


def test_observer_rejects_incomplete_ground_and_invalid_goals(tmp_path):
    descriptor=experiment()["ground"]
    path=tmp_path/"g.json"
    path.write_text(json.dumps(descriptor))
    for goal in ({"kind":"digest.sha256","steps":-1},
                 {"kind":"machine.execute","steps":10},
                 {"kind":"digest.sha256","steps":10,"max_work":4}):
        req=ObserverIntention("ground.navigate","workspace:g.json",
                             options={"goal":goal})
        result=arbitrate_orbit(req,root=tmp_path)
        assert result["status"] in ("REFUSED","UNRESOLVED")
    bad=ObserverIntention("ground.navigate","workspace:g.json",
                          options={"goal":{"steps":"1000","kind":"bytes.snapshot"}})
    assert arbitrate_orbit(bad,root=tmp_path)["status"]=="REFUSED"
    path.write_text('{"status":"tampered"}')
    req=ObserverIntention("ground.navigate","workspace:g.json",
                          options={"goal":{"steps":1,"kind":"state.scalar"}})
    assert arbitrate_orbit(req,root=tmp_path)["status"]=="UNRESOLVED"


def test_compressiveness_is_empirical_and_reconstruction_is_exact():
    r=experiment()["report"]
    assert r["status"]=="VERIFIED_ON_OBSERVED_AND_UNSEEN_SAMPLES"
    assert len(r["independent_unseen_frame_indices"])==5
    assert r["compressed_ratio"]>20
    assert r["ground_description_bytes"]<r["raw_training_and_holdout_bytes"]
    assert r["route"]==[
        "interpret-span","navigate-affine","materialize-span",
        "digest-realization"]
    assert r["prediction_scope"]=="empirically_heldout_not_universal_future_proof"
