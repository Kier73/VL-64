"""Challenge the portable model seed with an independent full-vocabulary reference.

For the prompt [42,75], compare two greedy steps over all 49,152 outputs and
two steps restricted to the ordered pair (9805,787), feeding each decision
back into its own trajectory. Seed and reference run in separate processes.

python -m experiments.model_fabric.model_seed_reference --seed artifacts/VL64-model.pyz \
    --snapshot LOCAL_SMOLLM_135M --out artifacts/reference.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

MODEL = "HuggingFaceTB/SmolLM-135M"
REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
PINS = {
    "model.safetensors": "c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093",
    "config.json": "a1fe6f43e20f7a6c6dbc6380222af9526b5cef262446391a281c038249e3e3b7",
}
PROMPT, PAIR = (42, 75), (9805, 787)
VOCAB, STEPS = 49152, 2
# Declared before running the seed, in the logit units of the CPU FP32 reference.
ABSOLUTE_TOLERANCE = 2e-4


def file_digest(path):
    """H(path)=SHA256(sequence of all file bytes)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def reference(snapshot):
    """Compute L_t=HF_FP32(x₁:ₜ), then extend with argmax_{k∈A} L_t[k] for each A."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    import transformers

    torch.set_num_threads(1)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="eager").cpu().eval()
    rows = {}
    with torch.inference_mode():
        for scope in ("full", "pair"):
            context = list(PROMPT)
            trajectory = []
            for _ in range(STEPS):
                logits = model(input_ids=torch.tensor([context]), use_cache=False).logits[0, -1]
                if logits.dtype != torch.float32 or logits.shape != (VOCAB,) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid reference logit vector")
                row = logits.numpy().copy()
                gap = float(row[PAIR[0]]) - float(row[PAIR[1]])
                if scope == "pair" and gap == 0:
                    raise ValueError("Reference has a restricted-pair tie on this trajectory")
                token = int(row.argmax()) if scope == "full" else PAIR[int(gap < 0)]
                trajectory.append((list(context), row, token))
                context.append(token)
            rows[scope] = trajectory
    software = {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": np.__version__,
                "device": "cpu", "dtype": "float32", "attention": "eager", "torch_threads": 1}
    return rows, software


def challenge(seed, snapshot, directory, scope):
    """Evaluate C(A,p,2) in an isolated interpreter; measure t_end−t_start for the whole call."""
    args = [sys.executable, "-I", "-S", str(seed), "run", "--snapshot", str(snapshot),
            "--tokens", *map(str, PROMPT), "--steps", str(STEPS)]
    args.extend(["--include-logits"] if scope == "full" else ["--pair", *map(str, PAIR)])
    start = perf_counter()
    run = subprocess.run(args, cwd=directory, check=True, stdout=subprocess.PIPE, text=True)
    elapsed = perf_counter() - start
    result = json.loads(run.stdout)
    if len(result["generated_ids"]) != STEPS or len(result["steps"]) != STEPS:
        raise ValueError("Seed returned incomplete trajectory")
    return result, elapsed


def compare(seed_result, trajectory, *, full):
    """Compare every matched prefix: e∞=maxᵢ|Cᵢ−Lᵢ|, e_RMS=√mean((C−L)²), and argmax."""
    actual_ids = seed_result["generated_ids"]
    if any(type(x) is not int or not 0 <= x < VOCAB for x in actual_ids):
        raise ValueError("Invalid seed token ID")
    if full and len(seed_result["logits"]) != STEPS:
        raise ValueError("Full seed output must supply every logit row")
    records = []
    for t, (context, reference_row, token) in enumerate(trajectory):
        same_context = context == [*PROMPT, *actual_ids[:t]]
        gap = float(reference_row[PAIR[0]]) - float(reference_row[PAIR[1]])
        record = {"reference_context": context, "same_context": same_context,
                  "reference_id": token, "seed_id": actual_ids[t],
                  "reference_yes_minus_no": gap, "reference_pair_sign": int(np.sign(gap)),
                  "passed": same_context and token == actual_ids[t]}
        if not full:
            if actual_ids[t] not in PAIR:
                raise ValueError("Restricted seed returned a token outside its observed pair")
            record["seed_pair_sign_from_id"] = 1 if actual_ids[t] == PAIR[0] else -1
        if full and same_context:
            candidate = np.asarray(seed_result["logits"][t], dtype="<f8")
            if candidate.shape != (VOCAB,) or not np.isfinite(candidate).all():
                raise ValueError("Expected one finite seed logit per vocabulary token")
            difference = candidate - reference_row.astype(np.float64)
            index = int(np.argmax(np.abs(difference)))
            largest, next_largest = np.sort(reference_row)[-2:][::-1]
            record.update({
                "logits_compared": VOCAB, "max_absolute_logit_error": float(abs(difference[index])),
                "max_error_token_id": index, "rms_logit_error": float(np.sqrt(np.mean(difference**2))),
                "reference_top1_minus_runner_up": float(largest) - float(next_largest),
                "reference_logits_f32le_sha256": hashlib.sha256(reference_row.astype("<f4").tobytes()).hexdigest(),
                "seed_logits_f64le_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
            })
            record["passed"] &= (int(candidate.argmax()) == actual_ids[t] and
                                 record["max_absolute_logit_error"] <= ABSOLUTE_TOLERANCE)
        records.append(record)
    return records


def main():
    """Accept iff all input hashes match, both greedy trajectories agree, and max_t e∞≤τ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    seed, snapshot = args.seed.resolve(), args.snapshot.resolve()
    if {name: file_digest(snapshot / name) for name in PINS} != PINS:
        raise ValueError("Checkpoint or configuration SHA-256 differs from the pinned source")
    trajectories, software = reference(snapshot)
    report = {
        "schema": "vl64.isolated-model-seed-reference.v1", "model": MODEL, "revision": REVISION,
        "source_sha256": PINS, "source_model_file_bytes": (snapshot / "model.safetensors").stat().st_size,
        "seed_sha256": file_digest(seed), "seed_bytes": seed.stat().st_size,
        "seed_python_flags": ["-I", "-S"], "reference_runtime": software,
        "input_token_ids": list(PROMPT), "ordered_pair_token_ids": list(PAIR),
        "full_vocabulary_size": VOCAB, "max_absolute_error_tolerance": ABSOLUTE_TOLERANCE,
    }
    with TemporaryDirectory() as empty_directory, ThreadPoolExecutor(max_workers=2) as workers:
        # Both observers run in separate processes. Their wall times overlap and are not a speed comparison.
        calls = {scope: workers.submit(challenge, seed, snapshot, empty_directory, scope)
                 for scope in ("full", "pair")}
        report["seed_process_timing"] = "Two concurrent isolated workers, wall time per complete CLI call"
        for scope, call in calls.items():
            actual, elapsed = call.result()
            checks = compare(actual, trajectories[scope], full=scope == "full")
            report[scope] = {"reference_checks": checks, "seed_subprocess_wall_seconds": elapsed,
                             "seed_receipt": {key: value for key, value in actual.items() if key != "logits"}}
    if file_digest(seed) != report["seed_sha256"]:
        raise ValueError("Seed archive changed during comparison")
    report["passed"] = all(c["passed"] for scope in ("full", "pair") for c in report[scope]["reference_checks"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.out)}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
