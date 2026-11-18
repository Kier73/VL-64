"""Compare independently verified Internet Ground captures over time.

A raw response can change while its useful interpreted state remains the
same. Two samples reveal observed differences, not predictive guarantees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .virtual_net import verify_dataset


def normalize(record):
    name = record["operation"]
    if name in {"github_commit_metadata", "github_branch_reference"}:
        return "github_default_branch_identity"
    return name


def observation_value(record, navigation):
    """Choose the smallest *observed* state relevant to each operation."""
    kind = normalize(record)
    if kind == "github_default_branch_identity":
        return {"commit": navigation.get("commit")}
    if kind.startswith("os_dns:"):
        return {"address_set": record.get("address_set_sha256"),
                "count": record.get("address_count")}
    if kind.startswith("cloudflare_doh_"):
        return {"address_set": record.get("answer_set_sha256"),
                "rcode": record.get("dns_rcode"),
                "count": record.get("answer_count")}
    if kind == "git_v2_capabilities":
        return {"capabilities": record.get("capabilities")}
    if kind == "git_v2_ls_refs":
        return {"head": record.get("head_commit")}
    if kind == "github_pinned_readme_conditional":
        return {"result": record.get("cache_result"),
                "http_status": record.get("http_status"),
                "bytes_avoided": record.get("bytes_avoided")}
    if kind == "tls_direct:example.com":
        return {"tls": record.get("tls_version"),
                "cipher": record.get("cipher"),
                "certificate": record.get("certificate_sha256")}
    if kind == "tcp_loopback_endpoint_transform":
        return {"input": record.get("input_sha256"),
                "output": record.get("output_sha256"),
                "status": record.get("status")}
    return {"body_sha256": record.get("body_sha256"),
            "truncated": record.get("truncated"),
            "http_status": record.get("http_status")}


def compare_captures(first, second):
    checks = [verify_dataset(first), verify_dataset(second)]
    if any(check["status"] != "VERIFIED" for check in checks):
        raise ValueError("Both datasets must independently pass replay checks")
    maps = [{normalize(rec): rec for rec in item["observations"]}
            for item in (first, second)]
    common = sorted(set(maps[0]) & set(maps[1]))
    comparisons = []
    for op in common:
        a, b = (m[op] for m in maps)
        left = observation_value(a, first["navigation"])
        right = observation_value(b, second["navigation"])
        if None in left.values() or None in right.values():
            classification = "INSUFFICIENT_INTERPRETATION"
        else:
            classification = "SAME_OBSERVED_STATE" if left == right else "CHANGED_OBSERVED_STATE"
        comparisons.append({
            "operation": op, "classification": classification,
            "before": left, "after": right,
            "raw_response_changed": (
                a.get("body_sha256") != b.get("body_sha256")
                if a.get("body_sha256") and b.get("body_sha256") else None),
            "elapsed_ms": [a.get("elapsed_ms"), b.get("elapsed_ms")],
        })
    classes = {key: sum(x["classification"] == key for x in comparisons)
               for key in ("SAME_OBSERVED_STATE", "CHANGED_OBSERVED_STATE",
                           "INSUFFICIENT_INTERPRETATION")}
    return {
        "schema": "vl64.internet-ground-temporal-comparison.v1",
        "first_dataset": first["dataset_digest"],
        "second_dataset": second["dataset_digest"],
        "source_commits": [first["navigation"].get("commit"),
                           second["navigation"].get("commit")],
        "shared_operations": len(common),
        "first_only": sorted(set(maps[0]) - set(maps[1])),
        "second_only": sorted(set(maps[1]) - set(maps[0])),
        "counts": classes, "comparisons": comparisons,
        "scopes": [
            "two_observed_sessions_not_a_statistical_predictability_claim",
            "semantic_views_are_operation_specific",
            "timing_is_diagnostic_not_an_invariant",
            "truncated_http_body_digest_is_a_prefix_only",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two verified Internet Ground snapshots")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out")
    args = parser.parse_args()
    first, second = (json.loads(Path(p).read_text(encoding="utf-8"))
                     for p in (args.first, args.second))
    report = compare_captures(first, second)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(output, encoding="utf-8")
    print("VL64_INTERNET_TEMPORAL_SUMMARY " + json.dumps({
        "counts": report["counts"],
        "source_commits": report["source_commits"],
        "shared_operations": report["shared_operations"],
        "raw_changed_but_interpreted_same": [
            x["operation"] for x in report["comparisons"]
            if x["classification"] == "SAME_OBSERVED_STATE"
            and x["raw_response_changed"] is True],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
