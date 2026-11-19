"""Token row from a qualified observer combined with genuine saved BF16 norm/Q bytes."""
import hashlib
import json
import math
import struct

import pytest

from experiments.model_fabric.remote_token_probe import (
    MODEL_GROUND, SLICE_GROUND, decode_bf16_exact, execute_real_token_q,
)
from experiments.model_fabric.execution_dag import compile_decoder_dag


def fixture():
    ground=json.loads(MODEL_GROUND.read_text())
    capsule=json.loads(SLICE_GROUND.read_text())
    plan=compile_decoder_dag(ground)
    values=[math.sin(i/23)*0.07 for i in range(plan["architecture"]["hidden"])]
    payload=b"".join(struct.pack("<H", struct.unpack("<I",struct.pack("<f",x))[0]>>16)
                     for x in values)
    calls=[]
    def host(request):
        calls.append(request)
        assert request["tensor"]=="model.embed_tokens.weight"
        assert request["byte_end_exclusive"]-request["byte_begin"]==1152
        return dict(request,payload=payload,observer_host="qualified-test-observer",
                    status="QUALIFIED_EXACT_RANGE",
                    sha256=hashlib.sha256(payload).hexdigest())
    return ground,capsule,host,calls,payload


def test_exact_pretrained_capsule_combines_with_observer_token_embedding():
    ground,capsule,host,calls,payload=fixture()
    receipt=execute_real_token_q(
        observed=ground,capsule=capsule,observer_host=host,
        approved_hosts={"qualified-test-observer"},token_id=42)
    assert receipt["status"]=="ACTUAL_WEIGHT_BACKED_TOKEN_EMBEDDING_AND_Q_ROWS_EXECUTED"
    assert receipt["fresh_embedding_bytes_observed"]==1152
    assert receipt["previously_admitted_norm_q_bytes"]==5760
    assert receipt["exact_range_count"]==1
    assert receipt["reference_passed"] is True
    assert len(receipt["first_four_q_coordinates"])==4
    assert receipt["full_model_forward_executed"] is False
    assert len(calls)==1
    assert calls[0]["byte_begin"]==ground["tensors"]["header_bytes"]+42*576*2
    assert len(decode_bf16_exact(payload))==576
    assert receipt["embedding_payload_sha256"]==hashlib.sha256(payload).hexdigest()
    assert receipt["receipt_digest"]==execute_real_token_q(
        observed=ground,capsule=capsule,observer_host=host,
        approved_hosts={"qualified-test-observer"},token_id=42)["receipt_digest"]


def test_token_embedding_refuses_false_host_identity():
    ground,capsule,host,_,_=fixture()
    with pytest.raises(ValueError,match="Unqualified"):
        execute_real_token_q(observed=ground,capsule=capsule,observer_host=host,
                             approved_hosts={"different-host"})


def test_token_embedding_fails_on_capsule_mismatch():
    ground,capsule,host,_,_=fixture()
    capsule=dict(capsule,model="other/model")
    with pytest.raises(ValueError,match="digest"):
        execute_real_token_q(observed=ground,capsule=capsule,observer_host=host,
                             approved_hosts={"qualified-test-observer"})


def test_bf16_nonfinite_is_rejected():
    with pytest.raises(ValueError,match="Nonfinite"):
        decode_bf16_exact(b"\xc0\x7f")
