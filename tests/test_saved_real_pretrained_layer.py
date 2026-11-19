"""Verify the persisted first REAL pretrained transformer-layer CI observation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.model_fabric.execution_dag import compile_decoder_dag,MAX_RANGE
from experiments.model_fabric.pretrained_layer import LAYER_TENSORS

ROOT=Path(__file__).resolve().parents[1]/"experiments/model_fabric/datasets"


def test_persisted_pretrained_layer0_receipt_matches_admitted_tensor_manifest():
    source=(ROOT/"2026-11-18-real-layer0-reference-receipt.json").read_bytes()
    # SHA-256 of the actual successful Actions artifact in run 44451577911.
    assert hashlib.sha256(source).hexdigest()==(
        "97f7b3c63b9372cc5576e3f0cf03d68bee5d36c5f53de2d18994b9ee92483da9")
    recorded=json.loads(source)
    observed=json.loads((ROOT/"2026-11-18-pinned-smollm2-tensor-ground.json").read_text())
    plan=compile_decoder_dag(observed)
    assert recorded["status"]=="REAL_BF16_LAYER0_INDEPENDENTLY_VERIFIED"
    assert recorded["model"]==plan["model"]
    assert recorded["revision"]==plan["revision"]
    assert recorded["dag_digest"]==plan["dag_digest"]
    assert recorded["tokens"]==[42,75]
    assert recorded["layer"]==0
    sizes=[]
    for suffix in LAYER_TENSORS:
        tensor=plan["tensors"]["model.layers.0."+suffix]
        assert tensor["dtype"]=="BF16"
        sizes.append(math.prod(tensor["shape"])*2)
    embedding=plan["tensors"]["model.embed_tokens.weight"]
    assert embedding["dtype"]=="BF16"
    payload=sum(sizes)+len(recorded["tokens"])*embedding["shape"][1]*2
    ranges=sum(math.ceil(size/MAX_RANGE) for size in sizes)+len(recorded["tokens"])
    assert recorded["weight_payload_bytes_acquired"]==payload==7_082_496
    assert recorded["exact_range_count"]==ranges==115
    assert recorded["independent_reference_max_abs_error"]<=recorded["reference_tolerance_abs"]
    assert len(recorded["last_hidden_first_8"])==8
    assert len(recorded["range_witness_chain_digest"])==64
