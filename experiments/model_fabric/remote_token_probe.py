"""Real pretrained token embedding -> RMSNorm -> selected Q rows, not full LLM.

A trusted ranged observer supplies a real BF16 token row; the two existing
pretrained tensor slices are independently replayed and re-used. No full
checkpoint, model library, tokenizer or synthetic input vector is needed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from pathlib import Path

from vld import StructuralLinearOperator, domain_digest, rms_norm

from .execution_dag import (QualifiedTensorReader, compile_decoder_dag,
                            huggingface_range_host)
from .slice_ground import replay_slice_ground

DATASETS = Path(__file__).parent / "datasets"
MODEL_GROUND = DATASETS / "2026-11-18-pinned-smollm2-tensor-ground.json"
SLICE_GROUND = DATASETS / "2026-11-18-pretrained-bf16-slice-ground.json"


def decode_bf16_exact(data):
    if not isinstance(data, bytes) or len(data) % 2:
        raise ValueError("Invalid BF16 tensor bytes")
    result = [struct.unpack("<f", struct.pack("<I", bits << 16))[0]
              for (bits,) in struct.iter_unpack("<H", data)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Nonfinite BF16 pretrained tensor")
    return result


def execute_real_token_q(*, token_id=42, observed=None, capsule=None,
                         observer_host=None, approved_hosts=None):
    """Execute pretrained token 42 into actual first four layer-0 Q coordinates.

    With the default host this requests exactly one 1152-byte pinned HTTPS
    weight range; remaining weights come from the existing admitted capsule.
    When injecting a test host, evidence scope remains the bytes it supplied.
    """
    if observed is None:
        observed = json.loads(MODEL_GROUND.read_text(encoding="utf-8"))
    if capsule is None:
        capsule = json.loads(SLICE_GROUND.read_text(encoding="utf-8"))
    plan = compile_decoder_dag(observed)
    replay_slice_ground(capsule)
    a = plan["architecture"]
    if (capsule.get("model") != plan["model"] or
            capsule.get("revision") != plan["revision"] or
            capsule.get("remote_ground_digest") != plan["model_ground_digest"] or
            capsule.get("tensor_header_ground_digest") != plan["tensor_ground_digest"] or
            capsule.get("hidden_width") != a["hidden"] or
            capsule.get("rms_eps") != a["eps"]):
        raise ValueError("Pretrained capsule does not describe the execution DAG")
    if observer_host is None:
        observer_host = huggingface_range_host(plan)
        approved_hosts = {"pinned-huggingface-https-range"}
    if not approved_hosts:
        raise ValueError("No observer-host qualification")
    observed_embedding_bytes = []

    def capture(request):
        response = observer_host(request)
        if isinstance(response, dict) and request["tensor"] == "model.embed_tokens.weight":
            observed_embedding_bytes.append(response.get("payload"))
        return response

    reader = QualifiedTensorReader(plan, capture, approved_hosts=approved_hosts)
    embedding = reader.embedding(token_id)
    if len(observed_embedding_bytes) != 1:
        raise ValueError("Unexpected embedding tensor range decomposition")
    independent_embedding = decode_bf16_exact(observed_embedding_bytes[0])
    if independent_embedding != embedding:
        raise RuntimeError("Independent pretrained token embedding mismatch")
    hidden = a["hidden"]
    if len(embedding) != hidden:
        raise ValueError("Truncated pretrained embedding")
    gamma = decode_bf16_exact(base64.b64decode(capsule["norm_bf16_b64"], validate=True))
    q = decode_bf16_exact(base64.b64decode(capsule["q_bf16_b64"], validate=True))
    if len(gamma) != hidden or len(q) != hidden * 4:
        raise ValueError("Invalid saved Q tensor slice")
    normalized = rms_norm(embedding, gamma, eps=a["eps"])
    operator = StructuralLinearOperator(
        rows=4, cols=hidden,
        sparse_delta={(i, j): q[i*hidden+j] for i in range(4)
                      for j in range(hidden) if q[i*hidden+j] != 0})
    calculated = operator.matvec(normalized)
    scale = 1/math.sqrt(math.fsum(x*x for x in independent_embedding)/hidden + a["eps"])
    reference = [math.fsum(q[i*hidden+j]*independent_embedding[j]*scale*gamma[j]
                           for j in range(hidden)) for i in range(4)]
    residual = max(abs(x-y) for x, y in zip(calculated, reference))
    if residual > 1e-6:
        raise RuntimeError("Pretrained token projection differs from scalar reference")
    receipt = {
        "schema": "vl64.real-pretrained-token-embedding-q.v1",
        "status": "ACTUAL_WEIGHT_BACKED_TOKEN_EMBEDDING_AND_Q_ROWS_EXECUTED",
        "model": plan["model"], "revision": plan["revision"],
        "token_id": token_id, "dag_digest": plan["dag_digest"],
        "tensor_ground_digest": plan["tensor_ground_digest"],
        "admitted_pretrained_slice_digest": capsule["ground_digest"],
        "fresh_embedding_bytes_observed": reader.bytes_read,
        "previously_admitted_norm_q_bytes": len(gamma)*2+len(q)*2,
        "exact_range_count": reader.ranges,
        "range_witness_chain_digest": reader.receipt_chain,
        "embedding_payload_sha256": hashlib.sha256(observed_embedding_bytes[0]).hexdigest(),
        "first_four_q_coordinates": [round(x, 9) for x in calculated],
        "reference_max_abs_error": round(residual, 12),
        "reference_passed": True,
        "full_model_forward_executed": False,
        "scope": "one_real_token_embedding_with_real_layer0_norm_and_first_four_q_rows",
    }
    receipt["receipt_digest"] = domain_digest(receipt["schema"], receipt)
    return receipt


def main():
    try:
        result = execute_real_token_q()
    except (ValueError, OSError, TimeoutError) as exc:
        result = {"status": "UNRESOLVED", "reason": type(exc).__name__,
                  "scope": "no_real_pretrained_token_result_without_qualified_range"}
    print("VL64_REAL_PRETRAINED_TOKEN_Q " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
