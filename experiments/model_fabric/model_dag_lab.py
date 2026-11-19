"""CLI for evidence-only pretrained DAG compilation or opt-in actual execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution_dag import compile_decoder_dag, execute_tokens, huggingface_range_host

PINNED = Path(__file__).parent / "datasets/2026-11-18-pinned-smollm2-tensor-ground.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PINNED)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--allow-remote-weights", action="store_true")
    args = parser.parse_args()
    observed = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = compile_decoder_dag(observed)
    if args.tokens is not None:
        if not args.allow_remote_weights:
            parser.error("Weighted execution requires explicit --allow-remote-weights")
        receipt = execute_tokens(
            plan, args.tokens, huggingface_range_host(plan),
            approved_hosts={"pinned-huggingface-https-range"})
        result = receipt
    else:
        result = {
            "status": "STRUCTURAL_DAG_COMPILED_NO_WEIGHT_VALUES_OBSERVED",
            "model": plan["model"],
            "revision": plan["revision"],
            "dag_digest": plan["dag_digest"],
            "nodes": len(plan["nodes"]),
            "admitted_tensors": len(plan["tensors"]),
            "architecture": plan["architecture"],
            "source_header_ground_digest": plan["tensor_ground_digest"],
        }
    serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
