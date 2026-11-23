"""MPF-001: VL-64 points at *model pages*, not model payloads.

Run with an E-CPU checkout in PYTHONPATH; E-CPU's HSS-002 HTTPSMetadataObserver
is the sole source-acquisition interface. It requests only model source
descriptions via the repository host's /api/models/ metadata endpoint.

Usage:
  python -m experiments.model_fabric.virtual_page_fleet_lab \
    --observers 128 --out artifacts/vl64-virtual-page-fleet.json

Source weight/shard/tokenizer/model binaries are never requested. Each
VL64 ObserverIntention becomes an independently recoverable structural
position and hash-linked receipt. "Running model" here means a verified
source-structural interpreter, not a neural forward pass.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

from .virtual_page_fleet import (
    HostedModelSource, VirtualModelPageFleet,
)

DEFAULT_PAGES = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main",
    "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main",
)


def observe_pages(page_urls: tuple[str, ...]):
    """Reuse original E-CPU observer, no VL64 weight-payload HTTP capability."""
    try:
        from mailcpu.external_virtual_structure import (
            HTTPSMetadataObserver, MetadataNavigator, compile_virtual_read,
        )
    except ImportError as exc:
        raise RuntimeError(
            "E-CPU HSS-002 source observer must be importable in PYTHONPATH"
        ) from exc
    if not 1 <= len(page_urls) <= 32 or len(set(page_urls)) != len(page_urls):
        raise ValueError("model page fleet requires 1..32 distinct URLs")
    observer = HTTPSMetadataObserver(timeout=20.0)
    navigator = MetadataNavigator(observer)
    sources = []
    for i, url in enumerate(page_urls):
        start = perf_counter_ns()
        original = navigator.discover(url)
        cold_ns = perf_counter_ns() - start
        grounded = HostedModelSource(
            original.to_dict(), expected_head=original.ground_digest,
        )
        # Two different independently authored systems must agree on both
        # the source address and the resulting executable structural law.
        for op, args in (
            ("source.catalogue", {}),
            ("source.architecture", {}),
        ):
            assert (grounded.position(op, args) ==
                    compile_virtual_read(original, op, args).to_dict())
        sources.append((f"m{i:03d}", grounded, url, cold_ns))
    return sources, observer


def run(page_urls: tuple[str, ...] = DEFAULT_PAGES, *,
        observers_per_model: int = 128, workers: int = 16) -> dict:
    """Measure M models x N observers with source structures held externally."""
    if (type(observers_per_model) is not int
        or not 1 <= observers_per_model <= 512
        or type(workers) is not int or not 1 <= workers <= 64):
        raise ValueError("observer and worker counts exceed benchmark policy")
    sources, metadata_observer = observe_pages(page_urls)
    summaries = []
    last_checkpoint = None
    for n in tuple(sorted(set((1, 8, 64, observers_per_model)))):
        if n > observers_per_model:
            continue
        fleet = VirtualModelPageFleet()
        for alias, source, _, _ in sources:
            fleet.register(alias, source)
        start = perf_counter_ns()
        tasks = []
        for alias, source, _, _ in sources:
            for i in range(n):
                actor = f"observer-{alias}-{i:04d}"
                fleet.open(actor, alias, host_id=f"vl64-host-{i%8}")
                tasks.append((actor, "source.catalogue", {}, "catalogue-0001"))
        opened_ns = perf_counter_ns() - start
        t0 = perf_counter_ns()
        receipts = fleet.observe_many(tasks, workers=workers)
        runtime_ns = perf_counter_ns() - t0
        if len(receipts) != len(tasks):
            raise AssertionError("some virtual observer positions were not admitted")
        groups: dict[str, list[dict]] = defaultdict(list)
        for receipt in receipts:
            groups[receipt["model"]].append(receipt)
            if receipt["model_payload_bytes"] != 0:
                raise AssertionError("model payload entered VL64 receipt")
        for alias, group in groups.items():
            if (len(group) != n or len({r["head"] for r in group}) != n
                or len({r["position_head"] for r in group}) != 1
                or len({r["source_head"] for r in group}) != 1):
                raise AssertionError("observer isolation or source Ground sharing differs")
        # A replacement host re-navigates the same description. No additional
        # source API calls are made per observer, model or handoff event.
        first = receipts[0]
        changed = fleet.rehome(
            first["observer"], "vl64-rehomed-host",
            expected_tip=first["head"],
        )
        resumed = fleet.observe(
            first["observer"], "source.architecture", {},
            request_id="post-rehome-0002", expected_tip=changed["head"],
        )
        if resumed["epoch"] != 1 or resumed["model_payload_bytes"] != 0:
            raise AssertionError("receiver host change lost source observer identity")
        checkpoint = fleet.snapshot()
        restored_start = perf_counter_ns()
        successor = VirtualModelPageFleet.restore(
            checkpoint, expected_digest=checkpoint["checkpoint_digest"],
        )
        replay_ns = perf_counter_ns() - restored_start
        if successor.snapshot() != checkpoint:
            raise AssertionError("described observation fleet failed independent replay")
        summaries.append({
            "models": len(sources), "observers_per_model": n,
            "total_observers": n * len(sources),
            "independent_observer_receipts": len(receipts),
            "independent_observer_heads": len({r["head"] for r in receipts}),
            "distinct_source_positions": len({r["position_head"] for r in receipts}),
            "source_ground_heads": [source.head for _,source,_,_ in sources],
            "new_model_metadata_observations_during_navigation": 0,
            "model_payload_bytes": 0,
            "model_payload_reads": 0,
            "observers_open_wall_ns": opened_ns,
            "warm_parallel_interpretation_ns": runtime_ns,
            "mean_observer_interpretation_ns": runtime_ns / len(receipts),
            "checkpoint_reconstruction_ns": replay_ns,
            "checkpoint_events": len(checkpoint["events"]),
            "handoff_accepted": True,
        })
        last_checkpoint = checkpoint

    # Orthogonal M:N test: one logical observer compares multiple sources,
    # rather than creating one independent observer per model.
    cross = VirtualModelPageFleet()
    for alias, source, _, _ in sources:
        cross.register(alias, source)
    cross.open_across_models("model-comparator", host_id="vl64-comparator")
    start_cross = perf_counter_ns()
    compared = cross.observe_across_models(
        "model-comparator", "source.architecture", {},
        request_id="compare-architectures", workers=workers,
    )
    cross_ns = perf_counter_ns() - start_cross
    if (len(compared) != len(sources)
        or len({receipt["source_head"] for receipt in compared.values()})
            != len(sources)
        or any(receipt["model_payload_bytes"] for receipt in compared.values())):
        raise AssertionError("cross-model observer lost source isolation")
    restored_cross = VirtualModelPageFleet.restore(
        cross.snapshot(), expected_digest=cross.snapshot()["checkpoint_digest"],
    )
    if restored_cross.snapshot() != cross.snapshot():
        raise AssertionError("multi-model observer causal state did not reconstruct")
    cross_model_summary = {
        "logical_observer": "model-comparator",
        "models_read": len(compared),
        "distinct_model_ground_heads": len({
            receipt["source_head"] for receipt in compared.values()
        }),
        "distinct_causal_heads": len({
            receipt["head"] for receipt in compared.values()
        }),
        "source_declared_model_types": {
            alias: receipt["position"]["result"]["observed_configuration"].get("model_type")
            for alias, receipt in sorted(compared.items())
        },
        "interpretation_wall_ns": cross_ns,
        "model_payload_bytes": 0,
    }
    return {
        "schema": "vl64.virtual-model-page-fleet-benchmark.v1",
        "source_role": "external_host_retains_model_data",
        "interpretation_role": "vl64_observer_intentions_over_ecpu_hss_structural_grounds",
        "source_descriptions": [
            {
                "model_alias": alias,
                "repository": source.repository,
                "revision": source.revision,
                "source_head": source.head,
                "page_url": url,
                "declared_source_bytes": source.declared_bytes,
                "declared_file_count": len(source.files),
                "declared_model_type": source.config.get("model_type"),
                "initial_metadata_observation_wall_ns": cold_ns,
            } for alias,source,url,cold_ns in sources
        ],
        "observed_source_model_bytes": sum(source.declared_bytes for _,source,_,_ in sources),
        "source_metadata_observations": metadata_observer.calls,
        "source_metadata_observed_bytes": metadata_observer.metadata_bytes,
        "model_payload_bytes_downloaded": 0,
        "model_payload_ranges_requested": 0,
        "model_payload_shards_copied": 0,
        "model_runtime_instances_started": 0,
        "source_local_trained_inference_calls": 0,
        "virtual_observer_interpreters_started": summaries[-1]["total_observers"],
        "source_interpretation_benchmarks": summaries,
        "single_observer_multiple_models": cross_model_summary,
        "final_checkpoint": last_checkpoint,
        "scope": "concurrent_virtual_source_structure_not_full_pretrained_neural_inference",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Navigate external model pages with many VL64 observers")
    parser.add_argument("--page", action="append", dest="pages",
                        help="Hugging Face model /tree/main page (repeat for many models)")
    parser.add_argument("--observers", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/vl64-virtual-model-page-fleet.json"))
    args = parser.parse_args(argv)
    report = run(tuple(args.pages) if args.pages else DEFAULT_PAGES,
                 observers_per_model=args.observers, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "final_checkpoint"},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
