"""Replay measured observer output summaries from the remote experiment."""
import json
from pathlib import Path


def test_live_observer_selection_summary_remains_explicit_and_weight_free():
    report = json.loads((Path(__file__).resolve().parents[1] /
        "experiments/observer_intent/datasets/"
        "2026-11-18-run-44389239409-summary.json").read_text())
    assert report["model_weight_payload_bytes_read"] == 0
    assert len(report["outputs"]) == 4
    assert all(x["plan_status"] == "ADMITTED_READ_ONLY_PLAN"
               for x in report["outputs"])
    assert report["constrained_architecture_selection"]["status"] == "OBSERVED_MODEL_STRUCTURE"
    selection = report["constrained_architecture_selection"]["selection"]
    assert selection["candidate_rank"] > 1
    assert selection["attempts"][0]["status"] == "PARTIAL_MODEL_STRUCTURE"
    assert all(x["weight_payload_bytes_read"] in (None, 0)
               for x in report["outputs"])
    assert any(x["format"] == "text" for x in report["outputs"])
    assert any(x["format"] == "jsonl" for x in report["outputs"])
