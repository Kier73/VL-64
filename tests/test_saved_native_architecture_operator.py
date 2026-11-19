"""Offline regression for archived pinned Qwen and DeepSeek operator receipts.

A CI log is not proof of full-checkpoint authenticity; this test verifies exact
evidence identity, documented scope, selected tensor families, and accounting.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from experiments.model_fabric.scaled_model_lab import PINNED

PATH=(Path(__file__).resolve().parents[1]/
      "experiments/model_fabric/datasets/2026-11-19-native-architecture-operator-observations.json")
HEX=re.compile(r"[0-9a-f]{64}\Z")

def test_archived_operator_receipts_model_revisions_and_ranges():
    record=json.loads(PATH.read_text(encoding="utf-8"))
    assert record["schema"]=="vl64.native-architecture-operator-observations.v1"
    assert record["observed_run_id"]==44547709486
    assert record["source_workflow_url"].endswith(str(record["observed_run_id"]))
    assert set(record["observations"])=={"qwen","deepseek"}
    expected={
        "qwen":("REAL_QWEN_GATED_DELTA_PARAMETERS_VERIFIED",20484,6,4),
        "deepseek":("REAL_DEEPSEEK_MXFP8_PROJECTION_ROW_VERIFIED",5280,7,2),
    }
    for family,report in record["observations"].items():
        status,nbytes,nranges,names=expected[family]
        assert (report["model"],report["revision"])==PINNED[family]
        assert report["status"]==status
        assert report["schema"]=="vl64.real-large-model-operator.v1"
        assert report["actual_weight_bytes_acquired"]==nbytes
        assert report["range_count"]==nranges
        assert report["max_abs_reference_error"]<=2e-5
        assert report["input_scope"]=="synthetic_hidden_vector_not_model_activation"
        assert "full layer" in report["scope"] or "no MoE layer" in report["scope"]
        assert len(report["tensor_names"])==names
        assert set(report["tensor_slice_sha256"])==set(report["tensor_names"])
        for k in ("index_sha256","index_ground_digest","range_witness_chain_digest"):
            assert HEX.fullmatch(report[k])
        assert all(HEX.fullmatch(d) for d in report["tensor_slice_sha256"].values())

def test_pretrained_real_results_not_overclaimed_as_full_generation():
    report=json.loads(PATH.read_text(encoding="utf-8"))["observations"]
    assert report["qwen"]["numeric_operator"]=="gated_delta_log_decay_and_beta_head0"
    assert 0<report["qwen"]["beta"]<1
    assert report["qwen"]["log_decay"]<=0
    assert report["deepseek"]["numeric_operator"]=="mxfp8_scaled_matrix_vector_row0"
    assert report["deepseek"]["tensor_names"]==[
        "layers.0.attn.wq_a.weight","layers.0.attn.wq_a.scale"]
    assert all("synthetic" in r["input_scope"] for r in report.values())
