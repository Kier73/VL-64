"""Regression fixture for two real Internet byte-fabric observers."""
import json
from pathlib import Path


def test_real_generic_observation_summary_and_cross_runtime_ground_identity():
    path = (Path(__file__).resolve().parents[1] /
            "experiments/byte_fabric/datasets/"
            "2026-11-18-run-44397602865-summary.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = {x["case"]:x for x in report["cases"]}
    assert len(cases) == 9
    assert cases["live-binary-png.json"]["bytes"] == 8991
    assert cases["live-binary-png.json"]["kind"] == "png"
    assert cases["live-page.json"]["kind"] == "html"
    assert cases["live-model-field.json"]["status"] == "EXECUTED_LOCAL"
    assert cases["live-partial.json"]["scope"] == "captured-prefix-only"
    assert cases["unknown-binary.json"]["kind"] == "opaque-bytes"
    assert report["cross_runtime_comparison"]["ground_digest_equal"] is True
    assert report["cross_runtime_comparison"]["compared_cases"] == 9
    assert all(x["ground_digest_equal"] for x in
               report["cross_runtime_comparison"]["cases"])
    assert report["claims"]["remote_model_inference_executed"] is False
