"""Non-affine finite closed orbits with exact compression and refusal evidence."""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
from vld import ObservedGround, ObserverIntention, domain_digest
from experiments.byte_fabric.discrete_ground import (
    discover_discrete_ground, DiscreteGroundNavigator,
)
from experiments.byte_fabric.ground_families import GroundFamilies
from experiments.byte_fabric.orbit_ground import ComposedGroundNavigator
from experiments.byte_fabric.orbit_lab import generate_observed_frames
from experiments.observer_intent.arbitrate import arbitrate, render_result


def fixtures(count=180, width=1024, offset=427):
    tokens=[hashlib.sha256(
        ("vl64-states-with-nonlinear-transitions/"+str(i)).encode()
    ).digest()[:2] for i in range(20)]
    assert len(set(tokens))==20
    template=(b"COMMUNICATION-IS-STATE-OBSERVATION|"*40)[:width]
    template=template[:offset]+b"\x00\x00"+template[offset+2:]
    prefix_len,period=3,17
    states=[tokens[i if i<prefix_len else
                   prefix_len+(i-prefix_len)%period] for i in range(count)]
    frames=[template[:offset]+state+template[offset+2:] for state in states]
    return frames,states,prefix_len,period


def trained():
    frames,tokens,start,period=fixtures()
    learned=discover_discrete_ground(frames[:58],frames[58:110])
    return learned,frames,tokens,start,period


def test_discover_nonaffine_ground_without_declaring_state_layout():
    learned,frames,tokens,start,period=trained()
    ground=learned["ground"]
    assert ground["variable_offset"]==427
    assert ground["variable_width"]==2
    assert ground["preperiod_states"]==start
    assert ground["cycle_states"]==period
    assert ground["claims"]["unobserved_future_behavior_proven"] is False
    assert ground["claims"]["closed_successor_relation_witnessed"]
    assert learned["measurement"]["heldout_exact_frames"]==52
    assert learned["measurement"]["compression_ratio"]>25
    interpreter=DiscreteGroundNavigator(ground)
    assert interpreter.check_against_core(frames[:58],10**12)
    for i in [0,1,2,3,19,20,21,57,58,75,109,110,135,179]:
        outcome=interpreter.realize(i,"bytes.snapshot",include_data=True)
        assert base64.b64decode(outcome["result"]["data_b64"])==frames[i]
        assert outcome["result"]["sha256"]==hashlib.sha256(frames[i]).hexdigest()
        assert [s["relation"] for s in outcome["stage_receipts"]]==[
            "interpret-token","navigate-closed-cycle","materialize-token"]


def test_arbitrarily_far_navigation_without_iteration():
    fitted,frames,tokens,start,period=trained()
    interpreter=DiscreteGroundNavigator(fitted["ground"])
    horizon=10**12
    result=interpreter.realize(horizon,"digest.sha256")
    assert result["status"]=="OBSERVED_GROUND_NAVIGATION"
    expected=frames[start+(horizon-start)%period]
    assert result["result"]["value"]==hashlib.sha256(expected).hexdigest()
    assert len(result["stage_receipts"])==4
    assert result["model_index"]==start+(horizon-start)%period
    assert result["navigation_complexity"].startswith("O(1)")


def test_competing_model_selection_prefers_finite_nonaffine_ground():
    frames,_,_,_=fixtures()
    library=GroundFamilies()
    selected=library.discover(frames[:58],frames[58:110])
    assert selected["selection"]["chosen_family"]=="closed-finite-cycle"
    assert selected["ground"]["schema"]=="vl64.discrete-orbit-ground.v1"
    assert selected["selection"]["status"]=="SELECTED_SHORTEST_HELDOUT_VALIDATED_DESCRIPTION"
    affine=next(c for c in selected["selection"]["candidates"] if c["candidate"]=="modular-affine")
    assert affine["status"]=="UNRESOLVED"
    assert library.open(selected["ground"]).realize(10**12,"state.token")["status"]=="OBSERVED_GROUND_NAVIGATION"


def test_competing_model_selection_retains_affine_ground_when_better():
    frames=generate_observed_frames(100)
    library=GroundFamilies()
    selected=library.discover(frames[:32],frames[32:80])
    assert selected["selection"]["chosen_family"]=="modular-affine"
    assert isinstance(library.open(selected["ground"]),ComposedGroundNavigator)
    assert selected["selection"]["description_bytes"]<2500
    assert library.open(selected["ground"]).check_against_core(10**12)


def test_new_observation_challenges_preserve_falsification():
    learned,frames,_,_,_=trained()
    interpreter=DiscreteGroundNavigator(learned["ground"])
    good=interpreter.test_new_observations([
        (115,frames[115]),(160,frames[160])])
    assert good["status"]=="CONSISTENT_ON_NEW_OBSERVATIONS"
    modified=bytearray(frames[160])
    modified[430] ^= 1  # a normally constant byte changes
    bad=interpreter.test_new_observations([
        (115,frames[115]),(160,bytes(modified))])
    assert bad["status"]=="CONTRADICTED"
    assert len(bad["contradictions"])==1
    assert bad["contradictions"][0]["index"]==160
    assert bad["observation_digest"]!=good["observation_digest"]
    assert learned["ground"]["ground_digest"]==interpreter.ground["ground_digest"]


def test_ambiguous_or_incomplete_observation_refused():
    frames,_,_,_=fixtures()
    with pytest.raises(ValueError,match="two observed cycles"):
        discover_discrete_ground(frames[:25],frames[25:40])
    altered=list(frames)
    altered[23]=altered[23][:427]+b"\xfe\xfe"+altered[23][429:]
    with pytest.raises(ValueError,match="cycle transitions conflict"):
        discover_discrete_ground(altered[:58],altered[58:110])
    altered=list(frames)
    modified=bytearray(altered[63])
    modified[95] ^= 1
    altered[63]=bytes(modified)
    with pytest.raises(ValueError,match="template changed"):
        discover_discrete_ground(altered[:58],altered[58:110])


def test_discrete_ground_rejects_untrusted_serialized_state():
    learned,frames,_,_,_=trained()
    raw=learned["ground"]
    for name,new_value in [
        ("variable_offset",4096),
        ("cycle_states",1),
        ("state_tokens_b64","AAAA"),
        ("schema","unrecognized"),
    ]:
        forged=dict(raw)
        forged[name]=new_value
        with pytest.raises(ValueError):
            DiscreteGroundNavigator(forged)
    forged=dict(raw)
    forged["cycle_states"]=10000
    forged["ground_digest"]=domain_digest("vl64.discrete-orbit-ground.v1",
            {k:v for k,v in forged.items() if k!="ground_digest"})
    with pytest.raises(ValueError):
        DiscreteGroundNavigator(forged)


def test_unknown_goal_and_invalid_horizon_refused():
    learned,_,_,_,_=trained()
    nav=DiscreteGroundNavigator(learned["ground"])
    assert nav.realize(1,"model.infer")["status"]=="UNRESOLVED"
    assert nav.realize(1,"digest.sha256",max_hops=3)["status"]=="UNRESOLVED"
    assert nav.realize(1,"digest.sha256",max_work=3)["status"]=="UNRESOLVED"
    assert nav.realize(-1,"state.token")["status"]=="REFUSED"
    assert nav.realize(10**20,"state.token")["status"]=="REFUSED"
    assert nav.realize(1,"bytes.snapshot",include_data="yes")["status"]=="REFUSED"


def test_registered_ground_driver_does_not_execute_unregistered_schemas():
    lib=GroundFamilies()
    with pytest.raises(ValueError):
        lib.open({"schema":"remote-code-eval","body":"danger"})
    with pytest.raises(ValueError):
        lib.register_driver("vl64.discrete-orbit-ground.v1",lambda x:x)
    with pytest.raises(ValueError):
        lib.register_builder("modular-affine",lambda t,v:t)


def test_discrete_navigator_supports_existing_external_observer_contract(tmp_path):
    learned,frames,_,_,_=trained()
    descriptor=tmp_path/"finite.json"
    descriptor.write_text(json.dumps(learned["ground"]))
    req=ObserverIntention(
        "ground.navigate","workspace:finite.json",output="json",
        options={"goal":{"kind":"digest.sha256","steps":10**12,
                         "max_work":4,"max_hops":4}})
    outcome=arbitrate(req,byte_root=tmp_path)
    assert outcome["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert outcome["navigation"]["result"]["value"]==hashlib.sha256(
        frames[3+(10**12-3)%17]).hexdigest()
    assert len(outcome["navigation"]["stage_receipts"])==4
    assert json.loads(render_result(outcome,"json"))["status"]=="OBSERVED_GROUND_NAVIGATION"
