"""Independent F32/BF16 one-layer, temporal and observer-evidence tests."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

import pytest

from vld import domain_digest
from experiments.model_fabric.execution_dag import (
    compile_decoder_dag,QualifiedTensorReader,
)
from experiments.model_fabric.pretrained_layer import (
    execute_pretrained_layer,PreparedLayer,LAYER_TENSORS,STAGES,_fingerprint,
)
from experiments.model_fabric.real_layer_lab import independent_reference
from test_execution_dag import (
    synthetic_pretrained_ground, qualified_host, scalar_reference,
)


@pytest.mark.parametrize("dtype",["BF16","F32"])
@pytest.mark.parametrize("tokens",[[4],[4,3,6]])
def test_complete_layer_matches_independent_scalar_reference(dtype,tokens):
    observed,data,weights=synthetic_pretrained_ground(dtype=dtype,layers=1)
    plan=compile_decoder_dag(observed)
    host=qualified_host(observed,data)
    ranges=defaultdict(list)
    def recording(request):
        response=host(request)
        ranges[request["tensor"]].append((request["byte_begin"],response["payload"]))
        return response
    actual=execute_pretrained_layer(
        plan,tokens,recording,approved_hosts={"local-f32-observer"})
    expected=independent_reference(plan,ranges,tokens)
    assert len(expected)==len(actual["output"])==8
    assert max(abs(x-y) for x,y in zip(expected,actual["output"]))<2e-6

    # Verify even the final (excluded) projection against the separate
    # full-decoder fixture, not the VL64 math operators.
    a=observed["architecture_fields"]
    final=actual["output"]
    gamma=weights["model.norm.weight"][1]
    scale=(math.fsum(x*x for x in final)/len(final)+a["rms_norm_eps"])**-0.5
    normalized=[x*scale*g for x,g in zip(final,gamma)]
    shape,w=weights["model.embed_tokens.weight"]
    projected=[math.fsum(w[row*8+j]*normalized[j] for j in range(8))
               for row in range(a["vocab_size"])]
    logits=scalar_reference(tokens,weights,a)
    assert max(abs(x-y) for x,y in zip(projected,logits))<2e-6
    assert all(set(x["stages"])=={"layer.0."+s for s in STAGES}
               for x in actual["positions"])
    assert actual["status"]=="WEIGHT_BACKED_PRETRAINED_LAYER_EXECUTED"
    assert actual["scope"].endswith("not_full_model_logits")
    size=2 if dtype=="BF16" else 4
    expected_bytes=(sum(math.prod(weights["model.layers.0."+s][0])
                        for s in LAYER_TENSORS)+len(tokens)*8)*size
    assert actual["weight_payload_bytes_acquired"]==expected_bytes
    assert actual["receipt_digest"]==execute_pretrained_layer(
        plan,tokens,qualified_host(observed,data),
        approved_hosts={"local-f32-observer"})["receipt_digest"]


def test_rope_positions_vary_and_kv_cache_is_causal():
    observed,data,_=synthetic_pretrained_ground(dtype="BF16")
    plan=compile_decoder_dag(observed)
    result=execute_pretrained_layer(plan,[4,3,4],qualified_host(observed,data),
                                    approved_hosts={"local-f32-observer"})
    states=result["positions"]
    assert states[0]["embedding"]==states[2]["embedding"]
    assert states[0]["stages"]["layer.0.q_rope"]!=states[2]["stages"]["layer.0.q_rope"]
    assert states[0]["stages"]["layer.0.attention"]!=states[1]["stages"]["layer.0.attention"]


def test_layer_reuses_admitted_weight_ranges():
    observed,data,_=synthetic_pretrained_ground(dtype="BF16")
    plan=compile_decoder_dag(observed)
    reader=QualifiedTensorReader(plan,qualified_host(observed,data),
                                 approved_hosts={"local-f32-observer"})
    runtime=PreparedLayer(reader)
    count=reader.ranges
    for _ in range(3):
        output,_=runtime.next([0.1]*plan["architecture"]["hidden"])
        assert len(output)==8
    assert reader.ranges==count
    assert len(runtime.keys)==3 and len(runtime.values)==3


def test_corrupt_weight_witness_refused():
    observed,data,_=synthetic_pretrained_ground()
    plan=compile_decoder_dag(observed)
    base=qualified_host(observed,data)
    def corrupt(request):
        response=base(request)
        if request["tensor"]=="model.layers.0.mlp.down_proj.weight":
            response["sha256"]="0"*64
        return response
    with pytest.raises(ValueError,match="Unqualified"):
        execute_pretrained_layer(plan,[2],corrupt,approved_hosts={"local-f32-observer"})


def test_recomputed_digest_cannot_authorize_unrecognized_graph_stage():
    observed,data,_=synthetic_pretrained_ground()
    plan=compile_decoder_dag(observed)
    changed=copy.deepcopy(plan)
    for node in changed["nodes"]:
        if node["id"]=="layer.0.gate":
            node["op"]="unadmitted_gate"
    changed["dag_digest"]=domain_digest(changed["schema"],
        {k:v for k,v in changed.items() if k!="dag_digest"})
    with pytest.raises(ValueError,match="stage differs"):
        execute_pretrained_layer(changed,[2],qualified_host(observed,data),
                                 approved_hosts={"local-f32-observer"})


def test_nonfinite_intermediate_refused():
    with pytest.raises(ValueError,match="Nonfinite"):
        _fingerprint([float("nan")])
