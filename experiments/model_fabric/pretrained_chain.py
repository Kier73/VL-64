"""Verified temporal handoffs across prepared pretrained Llama decoder layers.

The same portable Python interpreter currently executes each layer. Independent
approved observer-hosts may supply its exact weight ranges, but host labels do
not establish remote processor execution or cryptographic source authenticity.
Every handoff is an explicit bounded F64 payload, checked against the actual
upstream result before the downstream layer is advanced.
"""
from __future__ import annotations

import hashlib
import math
import struct

from vld import domain_digest
from .execution_dag import MAX_RANGE, QualifiedTensorReader
from .pretrained_layer import MAX_POSITIONS, PreparedLayer, _fingerprint

HANDOFF_SCHEMA = "vl64.pretrained-hidden-handoff.v1"
CHAIN_SCHEMA = "vl64.pretrained-layer-chain-receipt.v1"
GENESIS = "0" * 64


def _f64_bytes(values):
    if (not isinstance(values, (tuple, list)) or not values or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError("Nonfinite or malformed hidden vector")
    raw = struct.pack("<" + "d" * len(values), *values)
    if len(raw) > MAX_RANGE:
        raise ValueError("Hidden handoff exceeds one bounded observer frame")
    return raw


def make_handoff(plan, index, position, values, *, producer, range_chain,
                 preceding=GENESIS):
    """Prepare a data-bound, non-signed handoff for layer index -> index+1."""
    if (type(index) is not int or not 0 <= index < plan["architecture"]["layers"] - 1 or
            type(position) is not int or position < 0 or
            len(values) != plan["architecture"]["hidden"] or
            not isinstance(producer, str) or not producer or
            not isinstance(range_chain, str) or len(range_chain) != 64 or
            not isinstance(preceding, str) or len(preceding) != 64):
        raise ValueError("Unqualified layer handoff contract")
    raw = _f64_bytes(values)
    result = {
        "schema": HANDOFF_SCHEMA,
        "dag_digest": plan["dag_digest"],
        "tensor_ground_digest": plan["tensor_ground_digest"],
        "source_node": f"layer.{index}.ffn_residual",
        "target_node": f"layer.{index+1}.input_norm",
        "layer_from": index, "layer_to": index + 1,
        "position": position, "producer": producer,
        "source_weight_range_chain": range_chain,
        "predecessor_handoff_digest": preceding,
        "encoding": "ieee754-le-f64", "elements": len(values),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "payload": raw,
    }
    result["handoff_digest"] = domain_digest(
        HANDOFF_SCHEMA, {k: v for k, v in result.items()
                         if k not in ("payload", "handoff_digest")})
    return result


def admit_handoff(plan, handoff, *, index, position, preceding,
                  approved_producers, expected_vector=None):
    """Validate transport data, DAG edge, source assignment, and temporal order.

    A digest is only tamper-evidence relative to observed bytes, not a signature.
    When expected_vector is supplied, independently bind the received bytes to
    the actual upstream calculation (even if an attacker recomputed hashes).
    """
    if not isinstance(handoff, dict) or not isinstance(approved_producers, (set, tuple, list)):
        raise ValueError("Unqualified handoff object or producer policy")
    if (handoff.get("schema") != HANDOFF_SCHEMA or
            handoff.get("dag_digest") != plan["dag_digest"] or
            handoff.get("tensor_ground_digest") != plan["tensor_ground_digest"] or
            handoff.get("source_node") != f"layer.{index}.ffn_residual" or
            handoff.get("target_node") != f"layer.{index+1}.input_norm" or
            handoff.get("layer_from") != index or
            handoff.get("layer_to") != index + 1 or
            handoff.get("position") != position or
            handoff.get("producer") not in approved_producers or
            handoff.get("predecessor_handoff_digest") != preceding or
            handoff.get("encoding") != "ieee754-le-f64" or
            handoff.get("elements") != plan["architecture"]["hidden"] or
            not isinstance(handoff.get("source_weight_range_chain"), str) or
            len(handoff["source_weight_range_chain"]) != 64):
        raise ValueError("Handoff violates DAG, source or time contract")
    raw = handoff.get("payload")
    if (not isinstance(raw, bytes) or len(raw) != handoff["elements"] * 8 or
            len(raw) > MAX_RANGE or
            hashlib.sha256(raw).hexdigest() != handoff.get("payload_sha256")):
        raise ValueError("Handoff payload hash/length mismatch")
    fields = {k: v for k, v in handoff.items()
              if k not in ("payload", "handoff_digest")}
    if domain_digest(HANDOFF_SCHEMA, fields) != handoff.get("handoff_digest"):
        raise ValueError("Handoff evidence digest mismatch")
    values = list(struct.unpack("<" + "d" * handoff["elements"], raw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Handoff contains nonfinite hidden coordinates")
    if expected_vector is not None and raw != _f64_bytes(expected_vector):
        raise ValueError("Handoff differs from actual upstream output")
    return values


def execute_pretrained_chain(plan, tokens, range_hosts, *, approved_hosts,
                             compute_actors, approved_producers,
                             last_layer=1, transfer=None, include_output=True):
    """Compose layers 0..last_layer with causal state and verified handoffs.

    range_hosts and approved_hosts map -1 (embedding) and layer indices to
    host callback and allowed host IDs. compute_actors names local execution
    assignments, not remotely attested processors. transfer may transport or
    modify a handoff and always undergoes readmission before the next layer.
    Roll back all per-layer cache steps if any later stage fails at a position.
    """
    a = plan["architecture"]
    if (type(last_layer) is not int or not 1 <= last_layer < a["layers"] or
            not isinstance(tokens, (list, tuple)) or
            not 1 <= len(tokens) <= min(MAX_POSITIONS, a["context"]) or
            any(type(t) is not int or not 0 <= t < a["vocab"] for t in tokens) or
            not isinstance(range_hosts, dict) or not isinstance(approved_hosts, dict) or
            not isinstance(compute_actors, dict) or
            not isinstance(approved_producers, (set, list, tuple)) or
            (transfer is not None and not callable(transfer))):
        raise ValueError("Unqualified multi-layer execution request")
    for i in range(-1, last_layer + 1):
        if i not in range_hosts or i not in approved_hosts:
            raise ValueError("Missing layer/embedding observer-host assignment")
    for i in range(last_layer):
        if compute_actors.get(i) not in approved_producers:
            raise ValueError("Unapproved layer handoff producer")
    readers = {i: QualifiedTensorReader(plan, range_hosts[i],
                 approved_hosts=approved_hosts[i]) for i in range(-1, last_layer + 1)}
    layers = [PreparedLayer(readers[i], i) for i in range(last_layer + 1)]
    transcript = []
    previous_handoff = GENESIS
    for position, token in enumerate(tokens):
        prior_steps = [layer.step for layer in layers]
        pending = []
        try:
            state = readers[-1].embedding(token)
            initial = _fingerprint(state)
            for index, layer in enumerate(layers):
                state, stages = layer.next(state)
                if index < last_layer:
                    issued = make_handoff(
                        plan, index, position, state,
                        producer=compute_actors[index],
                        range_chain=readers[index].receipt_chain,
                        preceding=previous_handoff)
                    arrived = transfer(dict(issued)) if transfer is not None else issued
                    state = admit_handoff(
                        plan, arrived, index=index, position=position,
                        preceding=previous_handoff,
                        approved_producers=approved_producers,
                        expected_vector=state)
                    if arrived["source_weight_range_chain"] != readers[index].receipt_chain:
                        raise ValueError("Handoff does not bind source weight witness")
                    previous_handoff = arrived["handoff_digest"]
                    pending.append({
                        "from": index, "to": index + 1,
                        "digest": previous_handoff,
                        "payload_sha256": arrived["payload_sha256"],
                        "producer": arrived["producer"],
                        "source_weight_range_chain": arrived["source_weight_range_chain"],
                    })
                if index == last_layer:
                    end = _fingerprint(state)
            transcript.append({"position": position, "token_id": token,
                               "embedding": initial, "handoffs": pending,
                               "final_hidden": end})
        except Exception:
            for layer, old in zip(layers, prior_steps):
                layer.keys = layer.keys[:old]
                layer.values = layer.values[:old]
                layer.step = old
            # The containing caller must not reuse this chain after failure;
            # no receipt is committed. The current position is not published.
            raise
    receipt = {
        "schema": CHAIN_SCHEMA, "status": "WEIGHT_BACKED_PRETRAINED_LAYER_CHAIN_EXECUTED",
        "model": plan["model"], "revision": plan["revision"],
        "dag_digest": plan["dag_digest"],
        "tensor_ground_digest": plan["tensor_ground_digest"],
        "layers_executed": list(range(last_layer + 1)),
        "tokens": list(tokens), "positions": transcript,
        "weight_payload_bytes_acquired": sum(r.bytes_read for r in readers.values()),
        "exact_range_count": sum(r.ranges for r in readers.values()),
        "range_chains": {str(i): readers[i].receipt_chain for i in readers},
        "last_handoff_digest": previous_handoff,
        "output": state if include_output else _fingerprint(state),
        "scope": "selected_consecutive_pretrained_layers_not_full_vocabulary_logits",
        "host_scope": "qualified_weight_sources; locally_executed_layer_math_no_remote_compute_attestation",
    }
    receipt["receipt_digest"] = domain_digest(CHAIN_SCHEMA, receipt)
    return receipt
