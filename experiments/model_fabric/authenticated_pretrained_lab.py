"""MPF-003 opt-in live source-resident pretrained behavioral observer benchmark.

Requires an existing remote model executor, separately verified immutable
source-description Ground, independent operator/artifact pins, verified TLS
and a privately provisioned shared key. Never downloads a model or weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter_ns

from .authenticated_resident_host import AuthenticatedResidentHostClient
from .source_pinned_behavior_fleet import SourcePinnedBehaviorFleet
from .virtual_page_fleet import HostedModelSource, VirtualModelPageFleet


def run(*, source_record, source_head, operator_ground, artifact_head,
        endpoint, approved_host, secret, observers=4, workers=4,
        tokens=2, tokenizer_mode="plain_roles", tile_size=4096):
    if (type(observers) is not int or not 1 <= observers <= 128
            or type(workers) is not int or not 1 <= workers <= 32
            or type(tokens) is not int or not 1 <= tokens <= 24):
        raise ValueError("bounded source-resident benchmark policy required")
    source = HostedModelSource(source_record, expected_head=source_head)
    host = AuthenticatedResidentHostClient(
        endpoint=endpoint, secret=secret, source_head=source_head,
        ground_digest=operator_ground,
    )
    pages = VirtualModelPageFleet()
    pages.register("source", source)
    bridge = SourcePinnedBehaviorFleet(pages)
    actual = bridge.bind("source", host, approved_hosts={approved_host},
                         expected_artifact=artifact_head)
    if actual != operator_ground:
        raise ValueError("independently pinned operator Ground changed")
    for i in range(observers):
        name = f"causal-{i:04d}"
        pages.open(name, "source", host_id=f"receiver-{i%8}")
        bridge.open(name)

    rounds = []
    for turn, question in (
        ("first", "Explain the distinction between source structure and trained behavior."),
        ("second", "Give a concrete example that follows from the earlier answer."),
    ):
        jobs = [
            {"observer_id": f"causal-{i:04d}",
             "user_text": f"{question} Conversation {i}.",
             "request_id": f"turn-{turn}",
             "kind": "paragraph", "max_new_tokens": tokens,
             "tokenizer_mode": tokenizer_mode, "tile_size": tile_size}
            for i in range(observers)
        ]
        start = perf_counter_ns()
        receipts = bridge.turn_many(jobs, workers=workers)
        elapsed = perf_counter_ns()-start
        if (len({r["head"] for r in receipts}) != observers
                or len({r["source_head"] for r in receipts}) != 1
                or any(r["model_payload_bytes"] != 0 or
                       r["observation"]["remote_weight_bytes_received"] != 0
                       for r in receipts)):
            raise AssertionError("original-host behavior observer isolation failed")
        rounds.append({
            "turn": turn, "observers": observers, "wall_ns": elapsed,
            "computed_tokens": sum(len(r["observation"]["generated_tokens"])
                                   for r in receipts),
            "host_protocol_calls": sum(r["observation"]["host_calls"]
                                       for r in receipts),
            "independent_causal_receipts": len({r["head"] for r in receipts}),
            "receipt_heads": [r["head"] for r in receipts],
        })
    snapshot = bridge.snapshot()
    start = perf_counter_ns()
    recovered = SourcePinnedBehaviorFleet.restore(
        snapshot, expected_digest=snapshot["checkpoint_digest"])
    replay_ns = perf_counter_ns()-start
    if recovered.snapshot() != snapshot:
        raise AssertionError("authenticated original-host observer replay disagreed")
    return {
        "schema": "vl64.authenticated-original-pretrained-behavior-benchmark.v1",
        "model_source": source.repository,
        "immutable_source_revision": source.revision,
        "source_head": source.head,
        "operator_ground": operator_ground,
        "host_artifact_pin": artifact_head,
        "host_claimed_original_residency": True,
        "remote_host_protocol": "HMAC_SHA256_plus_HTTPS_exclusive_causal_lease",
        "scope": "authenticated_host_claim_and_complete_F32_logit_audit_not_independent_weight_attestation",
        "source_metadata_observations_during_inference": 0,
        "model_payload_bytes_downloaded": 0,
        "model_payload_bytes_copied_to_vl64": 0,
        "behavioral_rounds": rounds,
        "checkpoint_reconstruction_ns": replay_ns,
        "final_checkpoint": snapshot,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Observe actual behavior of an already-running original pretrained model")
    parser.add_argument("--source-description", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--operator-ground", required=True)
    parser.add_argument("--artifact-head", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--approved-host", required=True)
    parser.add_argument("--observers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--tokenizer-mode", choices=("plain_roles", "native_chat"),
                        default="plain_roles")
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/vl64-authenticated-pretrained-observers.json"))
    args = parser.parse_args(argv)
    secret = bytes.fromhex(os.environ["VL64_ORIGINAL_HOST_HMAC_HEX"])
    record = json.loads(args.source_description.read_text(encoding="utf-8"))
    report = run(
        source_record=record, source_head=args.source_head,
        operator_ground=args.operator_ground, artifact_head=args.artifact_head,
        endpoint=args.endpoint, approved_host=args.approved_host, secret=secret,
        observers=args.observers, workers=args.workers, tokens=args.tokens,
        tokenizer_mode=args.tokenizer_mode, tile_size=args.tile_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "final_checkpoint"},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
