"""Offline regression replay of the first real public Internet Ground capture."""
import json
from pathlib import Path

from experiments.internet_lab.virtual_net import verify_dataset


def test_saved_public_internet_dataset_replays_exact_evidence():
    data = Path(__file__).resolve().parents[1] / (
        "experiments/internet_lab/datasets/"
        "2026-11-18-github-run-44371133726.json")
    report = json.loads(data.read_text(encoding="utf-8"))
    verdict = verify_dataset(report)
    assert verdict["status"] == "VERIFIED", verdict
    assert verdict["packet_replays"] == 3
    assert verdict["graph_relations"] == 9
    assert report["summary"]["observations"] == 16
    assert report["summary"]["git_wire_rest_head_match"] is True
    assert report["navigation"]["hops"] == 3
    assert any(x.get("cache_result") == "NOT_MODIFIED_ACCEPTED_BY_ORIGIN"
               and x.get("bytes_avoided") == 6745
               for x in report["observations"])
