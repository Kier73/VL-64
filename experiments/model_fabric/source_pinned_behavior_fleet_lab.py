"""MPF-002 local trained-host concurrency benchmark (no external model payload reads).

This is an original count-trained byte LM fixture hosted once per experiment.
It validates the bridge's causal numerical protocol. Public page inference
requires a separately available original-host callable and artifact witness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from .byte_lm import fit_byte_lm
from .operational_ground import GROUND_SCHEMA, digest
from .operational_hosts import HostedTrainedByteLanguage
from .source_pinned_behavior_fleet import SourcePinnedBehaviorFleet
from .virtual_page_fleet import HostedModelSource, VirtualModelPageFleet, _hash


def run(*, observers=16, workers=8, tokens=3):
    if not 1 <= observers <= 128 or not 1 <= workers <= 32 or not 1 <= tokens <= 16:
        raise ValueError("bounded local trained-host benchmark required")
    training = (b"Source local trained computation is observed through short causal "
                b"messages. Many virtual observers address one resident model. ")*30
    learned = fit_byte_lm(training)
    record = {
        "schema": "mailcpu.external-virtual-structure.v1.source",
        "repository": "lab/mpf002-resident-byte-model",
        "revision": "a"*40,
        "files": [{"path": "config.json", "size_bytes": 512, "source_sha256": None}],
        "config": {"model_type": "count_trained_byte", "num_hidden_layers": 1},
        "observation_digests": ["b"*64],
        "scope": "external_structure_only_zero_model_payload",
    }
    source = HostedModelSource(record, expected_head=_hash(record))
    host = HostedTrainedByteLanguage(learned, observer_host="original-fixture-host")
    host.description["model"] = source.repository
    host.description["revision"] = source.revision
    host.bound_ground = digest(GROUND_SCHEMA, host.description)
    summaries = []
    for n in sorted(set((1, 8, observers))):
        if n > observers:
            continue
        pages = VirtualModelPageFleet()
        pages.register("fixture", source)
        bridge = SourcePinnedBehaviorFleet(pages)
        bridge.bind("fixture", host, approved_hosts={"original-fixture-host"},
                    expected_artifact=learned["ground_digest"])
        for i in range(n):
            observer = f"observer-{i:04d}"
            pages.open(observer, "fixture", host_id=f"receiver-{i%4}")
            bridge.open(observer)
        tasks = [
            {"observer_id": f"observer-{i:04d}",
             "user_text": f"Perform grounded trained continuation {i}.",
             "request_id": "causal-turn-0001", "max_new_tokens": tokens}
            for i in range(n)
        ]
        t0 = perf_counter_ns()
        events = bridge.turn_many(tasks, workers=workers)
        elapsed = perf_counter_ns()-t0
        checkpoint = bridge.snapshot()
        replay_start = perf_counter_ns()
        restored = SourcePinnedBehaviorFleet.restore(
            checkpoint, expected_digest=checkpoint["checkpoint_digest"])
        replay = perf_counter_ns()-replay_start
        assert restored.snapshot()==checkpoint
        assert all(ev["model_payload_bytes"]==0 for ev in events)
        summaries.append({
            "observers": n,
            "independent_causal_receipts": len({ev["head"] for ev in events}),
            "distinct_source_heads": len({ev["source_head"] for ev in events}),
            "distinct_operator_grounds": len({ev["operator_ground"] for ev in events}),
            "causal_tokens_computed": sum(
                len(ev["observation"]["generated_tokens"]) for ev in events),
            "host_protocol_calls": sum(ev["observation"]["host_calls"] for ev in events),
            "wall_ns": elapsed,
            "mean_wall_ns_per_observer": elapsed/n,
            "reconstruction_ns": replay,
            "host_model_copies": 0,
            "model_payload_bytes_downloaded": 0,
        })
    return {
        "schema": "vl64.source-pinned-behavior-benchmark.v1",
        "model": source.repository,
        "source_head": source.head,
        "operator_ground": host.bound_ground,
        "host_artifact_identity": learned["ground_digest"],
        "host_model_instances": 1,
        "trained_fixture_family": "count_trained_first_order_byte_lm",
        "operator": "vld.StructuralLinearOperator.matvec",
        "scope": "actual_trained_host_numeric_execution; synthetic_source_pin; not_original_HF_model_inference",
        "benchmarks": summaries,
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--observers", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tokens", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("artifacts/mpf002-trained-host-fleet.json"))
    args=p.parse_args(argv)
    report=run(observers=args.observers,workers=args.workers,tokens=args.tokens)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
