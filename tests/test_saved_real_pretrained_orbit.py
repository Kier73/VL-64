"""Verify the permanently retained real pretrained recovered-orbit CI receipt.

This tests the authentic historical Actions report's bytes and coherence with
its pinned observed tensor manifest, not a new inference without remote ranges.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from experiments.model_fabric.execution_dag import compile_decoder_dag, MAX_RANGE
from experiments.model_fabric.pretrained_layer import LAYER_TENSORS

DATA = Path(__file__).resolve().parents[1] / "experiments/model_fabric/datasets"


def test_recorded_real_bf16_orbit_v2_has_pinned_artifact_bytes_and_model_plan():
    raw = (DATA / "2026-11-18-real-pretrained-orbit-v2-receipt.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "29bcc7f9f35aced35da9ac40412bc5e3f6ce0b5a62640d29cb7ac1b140568fa9")
    result = json.loads(raw)
    assert result["schema"] == "vl64.real-pretrained-recovered-orbit.v2"
    assert result["status"] == "REAL_BF16_TWO_LAYER_RESTART_CONTINUITY_VERIFIED"
    assert result["tokens"] == [42,75]
    assert result["layers_executed"] == [0,1]
    assert (result["old_owner"],result["new_owner"]) == (
        "observer-assignment-a","observer-assignment-b")
    assert (result["old_epoch"],result["new_epoch"]) == (0,1)
    assert result["checkpoint_head"] != result["next_checkpoint_head"]
    for key in ("checkpoint_head","next_checkpoint_head","dag_digest"):
        assert re.fullmatch(r"[0-9a-f]{64}",result[key])
    assert result["independent_scalar_reference_max_abs_error"] <= result["tolerance_abs"]
    assert len(result["last_hidden_first_8"])==8
    assert result["compute_scope"].startswith("one_python_interpreter")
    assert "not_complete_model" in result["model_scope"]
    assert set(result["static_weight_identities"]) == {"0","1"}
    assert all(re.fullmatch(r"[0-9a-f]{64}",digest)
               for digest in result["static_weight_identities"].values())
    before=result["weight_range_witness_chains_before"]
    after=result["weight_range_witness_chains_after"]
    assert set(before)==set(after)=={"-1","0","1"}
    assert before!=after
    assert all(re.fullmatch(r"[0-9a-f]{64}",digest)
               for digest in list(before.values())+list(after.values()))

    manifest=json.loads((DATA/"2026-11-18-pinned-smollm2-tensor-ground.json").read_text())
    plan=compile_decoder_dag(manifest)
    assert result["dag_digest"] == plan["dag_digest"]
    assert result["model"] == plan["model"]
    assert result["revision"] == plan["revision"]
    sizes=[]
    for layer in (0,1):
        for suffix in LAYER_TENSORS:
            t=plan["tensors"][f"model.layers.{layer}."+suffix]
            assert t["dtype"] == "BF16"
            sizes.append(math.prod(t["shape"])*2)
    emb=plan["tensors"]["model.embed_tokens.weight"]
    assert emb["dtype"]=="BF16"
    # Each assignment loads the two layers once, and one fresh token row.
    per_assignment_bytes=sum(sizes)+emb["shape"][1]*2
    per_assignment_ranges=sum(math.ceil(n/MAX_RANGE) for n in sizes)+1
    assert per_assignment_bytes == 14_161_536
    assert per_assignment_ranges == 227
    assert result["acquired_BF16_bytes_by_assignment"] == [per_assignment_bytes]*2
    assert result["exact_range_reads_by_assignment"] == [per_assignment_ranges]*2
