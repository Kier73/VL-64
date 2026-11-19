"""Recheck preserved pinned two-layer Actions result and declared byte budget.

The stored report proves an observed historical CI result, not a new inference
without acquiring the underlying weight ranges again.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.model_fabric.execution_dag import MAX_RANGE, compile_decoder_dag
from experiments.model_fabric.pretrained_layer import LAYER_TENSORS

DATA = Path(__file__).resolve().parents[1] / "experiments/model_fabric/datasets"


def test_two_layer_ci_receipt_matches_saved_model_manifest_and_artifact_sha256():
    raw = (DATA / "2026-11-18-real-two-layer-pretrained-receipt.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "0123204cf81b58fd3c2e78772a704daba229f17f6cb2ca680f45614b964a916a")
    result = json.loads(raw)
    manifest = json.loads(
        (DATA / "2026-11-18-pinned-smollm2-tensor-ground.json").read_text())
    dag = compile_decoder_dag(manifest)
    assert result["status"] == "REAL_BF16_TWO_LAYER_CHAIN_INDEPENDENTLY_VERIFIED"
    assert result["dag_digest"] == dag["dag_digest"]
    assert result["model"] == dag["model"]
    assert result["revision"] == dag["revision"]
    assert result["tokens"] == [42, 75]
    assert result["layers"] == [0, 1]
    assert result["independent_reference_max_abs_error"] <= result["reference_tolerance_abs"]
    assert len(result["last_hidden_first_8"]) == 8
    assert result["compute_scope"] == (
        "single_python_interpreter_with_qualified_https_weight_ranges")
    assert "not_full_model" in result["scope"]

    sizes = []
    for index in (0, 1):
        for suffix in LAYER_TENSORS:
            tensor = dag["tensors"][f"model.layers.{index}." + suffix]
            assert tensor["dtype"] == "BF16"
            sizes.append(math.prod(tensor["shape"]) * 2)
    emb = dag["tensors"]["model.embed_tokens.weight"]
    assert emb["dtype"] == "BF16"
    token_bytes = len(result["tokens"]) * emb["shape"][1] * 2
    assert result["weight_payload_bytes_acquired"] == sum(sizes) + token_bytes == 14_162_688
    expected_ranges = sum(math.ceil(size / MAX_RANGE) for size in sizes)
    expected_ranges += len(result["tokens"])
    assert result["exact_range_count"] == expected_ranges == 228
    assert set(result["range_chains"]) == {"-1", "0", "1"}
    assert all(len(digest) == 64 for digest in result["range_chains"].values())
    assert len(result["layer_chain_receipt_digest"]) == 64
    assert len(result["last_handoff_digest"]) == 64
