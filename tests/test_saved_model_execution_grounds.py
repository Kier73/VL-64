"""Replay actual saved model Grounds, independently of remote network access."""
import json
from pathlib import Path

from experiments.model_fabric.byte_lm import ByteLanguageGround
from experiments.model_fabric.tiny_attention import (
    TinyCausalAttentionGround,deterministic_local_neural_weights
)
from experiments.model_fabric.slice_ground import replay_slice_ground
from experiments.model_fabric.model_seed import build_model_seed
from experiments.model_fabric.tensor_ground import observe_safetensors_header


DATA=Path(__file__).resolve().parents[1]/"experiments/model_fabric/datasets"


def stored(name):
    return json.loads((DATA/name).read_text(encoding="utf-8"))


def test_saved_trained_language_ground_generates_without_training_corpus():
    saved_model=stored("2026-11-18-trained-byte-language-ground.json")
    report=stored("2026-11-18-model-execution-evidence.json")
    model=ByteLanguageGround(saved_model)
    generated=model.generate(b"Virtual Layer",steps=48)
    assert saved_model["ground_digest"]==report["trained_language"][
        "trained_language_ground_digest"]
    assert generated["generated_hex"]==report["trained_language"][
        "greedy_generated_hex"]
    assert report["trained_language"]["heldout_byte_cross_entropy"]<8
    assert generated["status"]=="MODELED_WEIGHT_BACKED_INFERENCE"


def test_saved_tiny_neural_ground_fingerprint_matches_actual_weight_bytes():
    declared=stored("2026-11-18-tiny-attention-ground.json")
    actual=TinyCausalAttentionGround(deterministic_local_neural_weights())
    assert declared==actual.ground
    assert actual.next_token([0,1,2,3])[
        "status"]=="ACTUAL_LOCAL_WEIGHT_BACKED_NEURAL_INFERENCE"
    assert declared["training_claim"]=="seeded_untrained_numeric_example"


def test_saved_huggingface_manifest_describes_272_tensors_but_not_weight_values():
    hf=stored("2026-11-18-pinned-smollm2-tensor-ground.json")
    report=stored("2026-11-18-model-execution-evidence.json")
    assert hf["schema"]=="vl64.remote-model-topology-ground.v1"
    assert hf["model"]=="HuggingFaceTB/SmolLM2-135M"
    assert hf["tensor_header_status"]=="OBSERVED_COMPLETE_TENSOR_HEADER"
    assert hf["tensors"]["tensor_count"]==272
    assert hf["tensors"]["header_bytes"]==30536
    assert hf["tensors"]["declared_payload_bytes"]==269030016
    assert hf["architecture_fields"]["num_hidden_layers"]==30
    assert hf["architecture_fields"]["hidden_size"]==576
    assert hf["tensors"]["weight_payload_bytes_observed"]==0
    assert hf["model_execution_status"]=="NOT_EXECUTED_WEIGHTS_NOT_ADMITTED"
    assert hf["ground_digest"]==report["pinned_hugging_face_model"].get(
        "ground_digest",hf["ground_digest"])
    assert (DATA/"2026-11-18-pinned-smollm2-tensor-ground.json").stat().st_size<65536


def test_saved_pretrained_weight_slice_replays_without_any_http_connection():
    capsule=stored("2026-11-18-pretrained-bf16-slice-ground.json")
    report=stored("2026-11-18-model-execution-evidence.json")
    replay=replay_slice_ground(capsule)
    expected=report["pretrained_operation"]
    assert replay["status"]=="VERIFIED_LOCAL_REPLAY_OF_PRETRAINED_TENSOR_SLICE"
    assert replay["output_coordinates"]==expected["output_coordinates"]
    assert replay["receipt_digest"]==expected["portable_replay_receipt_digest"]
    assert replay["ground_digest"]==expected["portable_slice_ground_digest"]
    assert capsule["revision"]==report["pinned_hugging_face_model"]["revision"]
    assert capsule["training_claim"]=="actual_pinned_pretrained_model_slice_not_synthetic_weights"
    assert (DATA/"2026-11-18-pretrained-bf16-slice-ground.json").stat().st_size<65536


def test_portable_seed_embeds_the_exact_saved_real_model_slice():
    portable,report=build_model_seed()
    witness=stored("2026-11-18-pretrained-bf16-slice-ground.json")
    assert report["model_slice_ground_digest"]==witness["ground_digest"]
    assert report["model_seed_bytes"]<=65536
    assert report["core_source_bytes"]==65093
    assert report["complete_pretrained_model_execution"] is False
