"""Evidence-backed compression and navigation from exact byte observations."""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
from vld import domain_digest, affine_navigate
from experiments.byte_fabric.orbit_ground import (
    SpanInterpreter, infer_ground, discover_ground, ComposedGroundNavigator,
)


def frames(count=130, *, a=1664525, b=1013904223, modulus=2**32,
           codec=None, size=1024):
    codec = codec or SpanInterpreter(319, 4, "u-le")
    template = bytearray((b"VL64-GROUND-REUSE/" * ((size // 18) + 2))[:size])
    template[codec.offset:codec.offset+codec.width] = b"\0" * codec.width
    state = 7
    results = []
    states = []
    for _ in range(count):
        result = bytes(template[:codec.offset] + codec.encode(state) +
                       template[codec.offset+codec.width:])
        results.append(result)
        states.append(state)
        state = (a*state + b)%modulus
    return results, states, codec


def observed_ground(*, num=130):
    seen, states, codec = frames(num)
    fitted = infer_ground(seen[:30], seen[30:70],codec,2**32)
    return fitted,seen,states


def test_compressive_ground_reconstructs_all_witnessed_and_heldout_frames():
    sample, seen, states = observed_ground()
    ground = sample["ground"]
    navigator = ComposedGroundNavigator(ground)
    assert ground["a"] == 1664525
    assert ground["b"] == 1013904223
    assert sample["measurement"]["heldout_frames_checked"] == 40
    assert sample["measurement"]["future_states_proven"] is False
    assert sample["measurement"]["ratio_source_to_descriptor"] > 20
    for index in (0,1,2,13,29,30,45,69):
        result = navigator.realize(index,"bytes.snapshot",include_data=True)
        assert result["status"] == "OBSERVED_GROUND_NAVIGATION"
        assert result["result"]["sha256"] == hashlib.sha256(seen[index]).hexdigest()
        assert base64.b64decode(result["result"]["data_b64"]) == seen[index]
        assert [x["relation"] for x in result["stage_receipts"]] == [
            "interpret-span", "navigate-affine", "materialize-span"]
        assert result["ground_digest"] == ground["ground_digest"]


def test_navigation_skips_a_trillion_steps_and_reuses_power_basis():
    record, seen, states = observed_ground()
    machine = ComposedGroundNavigator(record["ground"])
    horizon = 10**12
    first = machine.realize(horizon, "state.scalar")
    assert first["status"] == "OBSERVED_GROUND_NAVIGATION"
    assert first["result"]["value"] == affine_navigate(
        record["ground"]["a"],record["ground"]["b"],states[0],horizon,2**32)
    after = len(machine._powers)
    second = machine.realize(horizon+1007,"digest.sha256")
    assert len(machine._powers) >= after
    assert second["route"]["estimated_work"] == 5
    assert [r["name"] for r in second["route"]["stages"]] == [
        "interpret-span","navigate-affine","materialize-span",
        "digest-realization"]
    assert machine.check_against_core(horizon)
    assert machine.check_against_core(horizon+1007)
    assert record["measurement"]["source_snapshot_bytes"] == 70*1024


def test_compressed_ground_stores_a_template_and_not_full_stream():
    record, seen, _ = observed_ground()
    desc = record["ground"]
    serialized = json.dumps(desc,sort_keys=True).encode()
    assert len(serialized) < 70*1024/20
    assert len(base64.b64decode(desc["template_b64"])) == 1024
    assert "witnesses" not in desc
    assert desc["training_evidence_digest"] != desc["heldout_evidence_digest"]


def test_refuse_nonconstant_template_and_heldout_counterexample():
    seen,_,codec=frames(40)
    tampered=list(seen)
    tampered[15]=tampered[15][:18] + b"X" + tampered[15][19:]
    with pytest.raises(ValueError,match="outside interpreter"):
        infer_ground(tampered[:25],tampered[25:35],codec,2**32)
    tampered=list(seen)
    offset=codec.offset
    raw=bytearray(tampered[28])
    raw[offset] ^= 3
    tampered[28]=bytes(raw)
    with pytest.raises(ValueError,match="Held-out"):
        infer_ground(tampered[:20],tampered[20:34],codec,2**32)


def test_ambiguous_even_states_do_not_fabricate_a_unique_law():
    codec=SpanInterpreter(8,1,"u-le")
    raw=bytearray(b"fixed=" + b"\0"*18)
    states=[2,4,8,16,32,64,128,0,0]
    samples=[]
    for state in states:
        arr=bytearray(raw)
        arr[8]=state
        samples.append(bytes(arr))
    with pytest.raises(ValueError,match="unique|independent"):
        infer_ground(samples[:6],samples[6:],codec,256)


def test_unknown_non_affine_progression_is_refused():
    codec=SpanInterpreter(4,2,"u-be")
    state=3
    sample=[]
    for _ in range(15):
        sample.append(b"mark" + codec.encode(state) + b"end")
        state = (state*state+3)%65521
    with pytest.raises(ValueError):
        infer_ground(sample[:10],sample[10:],codec,65521)


def test_ascii_decimal_and_big_endian_lenses_are_architecture_neutral():
    for interpreter, modulus in (
            (SpanInterpreter(1,5,"ascii-decimal"),10**5),
            (SpanInterpreter(1,2,"u-be"),65536)):
        state=7
        frameset=[]
        for i in range(22):
            frameset.append(b"[" + interpreter.encode(state) + b"]constant")
            state=(5*state+1)%modulus
        ground = infer_ground(frameset[:12],frameset[12:18],interpreter,modulus)
        nav=ComposedGroundNavigator(ground["ground"])
        assert nav.realize(17,"bytes.snapshot",include_data=True)["result"]["sha256"] == hashlib.sha256(frameset[17]).hexdigest()


def test_corrupted_ground_metadata_fails_validation():
    record,_,_=observed_ground()
    desc=record["ground"]
    for mutated in ("a","template_b64","modulus","initial_state"):
        fake=dict(desc)
        fake[mutated] = 0 if mutated != "template_b64" else "AA=="
        with pytest.raises(ValueError,match="digest"):
            ComposedGroundNavigator(fake)
    fake=dict(desc)
    fake["modulus"]=0
    fake["ground_digest"]=domain_digest("vl64.compressive-orbit-ground.v1",
                               {k:v for k,v in fake.items() if k!="ground_digest"})
    with pytest.raises(ValueError):
        ComposedGroundNavigator(fake)


def test_path_ceiling_and_unknown_target_do_not_run():
    rec,_,_=observed_ground()
    nav=ComposedGroundNavigator(rec["ground"])
    assert nav.realize(12,"digest.sha256",max_work=4)["status"]=="UNRESOLVED"
    assert nav.realize(12,"digest.sha256",max_hops=3)["status"]=="UNRESOLVED"
    assert nav.realize(12,"machine.execute")["status"]=="UNRESOLVED"
    assert nav.realize(-1,"bytes.snapshot")["status"]=="REFUSED"
    assert nav.realize(10**22,"bytes.snapshot")["status"]=="REFUSED"


def test_exact_trace_reconstruction_does_not_prove_unobserved_future():
    desc, _, _ = observed_ground()
    assert desc["ground"]["scope"]=="single-fixed-template-affine-observed-hypothesis"
    outcome=ComposedGroundNavigator(desc["ground"]).realize(10**12)
    assert outcome["interpretation_scope"] == (
        "witnessed_rule_projection_not_proven_future_system_behavior")


def test_discover_endian_and_span_without_prior_byte_offset():
    samples, states, interpreter=frames(80)
    observed=discover_ground(samples[:30],samples[30:65])
    chosen=observed["discovery"]["selected"]
    assert chosen==interpreter.to_dict()
    assert observed["discovery"]["status"]=="UNIQUE_OBSERVED_CODEC_HYPOTHESIS"
    assert chosen["offset"]==319
    assert len(observed["discovery"]["changing_positions"])==4
    assert any(x["encoding"]=="u-be" for x in
               observed["discovery"]["rejected_candidates"])
    nav=ComposedGroundNavigator(observed["ground"])
    assert nav.realize(64,"bytes.snapshot")["result"]["sha256"]==hashlib.sha256(samples[64]).hexdigest()


def test_discovery_refuses_independent_moving_fields_and_ambiguous_views():
    samples, _, codec=frames(25)
    two_fields=[]
    for i,frame in enumerate(samples):
        row=bytearray(frame)
        row[600]=i
        two_fields.append(bytes(row))
    with pytest.raises(ValueError,match="changing fields"):
        discover_ground(two_fields[:15],two_fields[15:22])
    stationary=[b"same-data"*15]*9
    with pytest.raises(ValueError,match="No witnessed changing"):
        discover_ground(stationary[:5],stationary[5:9])
