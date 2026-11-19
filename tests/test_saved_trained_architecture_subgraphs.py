"""Permanent structural cross-check of actual CI-observed Qwen and DeepSeek trained subgraphs."""
import json
import math
from pathlib import Path

RECEIPT=Path(__file__).resolve().parents[1]/"experiments/model_fabric/datasets/2026-11-19-qwen-deepseek-trained-subgraphs.json"


def test_saved_pinned_trained_large_model_subgraphs_are_not_full_inference():
    r=json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert r["schema"]=="vl64.saved-large-trained-subgraph-evidence.v1"
    assert r["source_actions_run"]==44556312995
    q=r["qwen"]; d=r["deepseek"]
    assert q["model"]=="Qwen/Qwen3.8-27B"
    assert q["revision"]=="37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    assert q["status"]=="REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED"
    assert q["value_head"]==0 and q["range_count"]==100
    assert q["weight_bytes"]==5277188
    assert q["max_abs_independent_error"]<=2e-5
    assert q["input_scope"]=="synthetic_hidden" and "not a full Qwen layer" in q["scope"]
    assert d["model"]=="deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert d["revision"]=="1bae0e1088993b9b82de08f59de5be5a34e42170"
    assert d["status"]=="REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED"
    assert d["selected_experts"]==[240,360,2,34,164,199]
    assert d["range_count"]==70 and d["weight_bytes"]==3933696
    assert d["input_scope"]=="synthetic_hidden"
    assert len(d["selected_weight_paths"])==6
    assert math.isclose(sum(d["routing_weights"]),1.5,rel_tol=0,abs_tol=1e-9)
    selected=d["selected_fp4_slices"]
    assert selected["status"]=="REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED"
    assert selected["selected_experts"]==d["selected_experts"]
    assert [x["expert"] for x in selected["slices"]]==d["selected_experts"]
    assert len(selected["slices"])==6
    assert selected["weight_bytes"]-d["weight_bytes"]==39984
    assert selected["range_count"]-d["range_count"]==36
    assert selected["max_abs_independent_error"]<=2e-5
    assert "not full MoE" in selected["scope"]
    for item in (q,d,selected):
        assert len(item["range_chain"])==64
