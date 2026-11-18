"""Compare evidence-backed public Internet sessions without assuming invariance."""
import json
from pathlib import Path

from experiments.internet_lab.compare_ground import compare_captures


def test_two_real_sessions_distinguish_state_and_representation_changes():
    root = Path(__file__).resolve().parents[1] / "experiments/internet_lab/datasets"
    first = json.loads((root / "2026-11-18-github-run-44371133726.json").read_text())
    second = json.loads((root / "2026-11-18-github-run-44372989749.json").read_text())
    compared = compare_captures(first, second)
    rows = {x["operation"]: x for x in compared["comparisons"]}
    assert compared["shared_operations"] >= 14
    assert compared["source_commits"][0] != compared["source_commits"][1]
    assert rows["github_default_branch_identity"]["classification"] == "CHANGED_OBSERVED_STATE"
    assert rows["git_v2_ls_refs"]["classification"] == "CHANGED_OBSERVED_STATE"
    assert rows["github_pinned_readme"]["classification"] == "SAME_OBSERVED_STATE"
    assert rows["github_pinned_research_goals"]["classification"] == "CHANGED_OBSERVED_STATE"
    assert rows["github_pinned_readme_conditional"]["classification"] == "SAME_OBSERVED_STATE"
    assert rows["cloudflare_doh_wire_A_example.com"]["classification"] == "SAME_OBSERVED_STATE"
    assert rows["cloudflare_doh_wire_A_example.com"]["raw_response_changed"] is True
    assert rows["cloudflare_doh_A_example.com"]["raw_response_changed"] is True
    assert compared["counts"]["SAME_OBSERVED_STATE"] > 0
    assert compared["counts"]["CHANGED_OBSERVED_STATE"] > 0
