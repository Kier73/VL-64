"""Sparse interacting byte-span Grounds and evidence-limited navigation."""
from __future__ import annotations

import base64
import hashlib

import pytest
from vld import domain_digest
from experiments.byte_fabric.multifield_ground import (
    MultiFieldNavigator, discover_multifield_ground,
)


TEMPLATE=(b"VL64-SELF-GROUND-OBSERVATION-CARRIER|"*20)[:512]
OFFSET_MODE=73
OFFSET_COUNTER=311


def frame(mode,count):
    original=bytearray(TEMPLATE)
    original[OFFSET_MODE]=mode
    original[OFFSET_COUNTER:OFFSET_COUNTER+2]=bytes([count,255-count])
    return bytes(original)


def independent_step(mode,count,action):
    if action=="tick":
        return mode,(count+(1 if mode==0 else 2))%4
    if action=="toggle":
        return 1-mode,count
    raise ValueError("unqualified independent action")


def fixtures(skip=()):
    training=[]
    for mode in (0,1):
        for counter in range(4):
            for action in ("tick","toggle"):
                if (mode,counter,action) in skip:
                    continue
                end_mode,end_counter=independent_step(mode,counter,action)
                training.append((frame(mode,counter),action,
                                 frame(end_mode,end_counter)))
    # Later independent observations from the same reference machine.
    heldout=list(training)
    return training,heldout


def fitted(skip=()):
    training,heldout=fixtures(skip)
    return discover_multifield_ground(training,heldout),training


def test_discovers_two_disjoint_interacting_fields_and_exact_template():
    result,training=fitted()
    ground=result["ground"]
    spans=ground["variable_spans"]
    assert spans==[
        {"offset":OFFSET_MODE,"width":1},
        {"offset":OFFSET_COUNTER,"width":2}]
    assert ground["frame_length"]==512
    assert result["measurement"]["fields"]==2
    assert result["measurement"]["states"]==8
    assert result["measurement"]["actions"]==2
    assert result["measurement"]["observed_edges"]==16
    assert result["measurement"]["unobserved_action_edges"]==0
    assert result["measurement"]["verified_heldout_count"]==16
    assert result["measurement"]["description_ratio"]>2
    navigator=MultiFieldNavigator(ground)
    for mode in (0,1):
        for count in range(4):
            state=bytes([mode,count,255-count])
            assert navigator.materialize(navigator.ids[state])==frame(mode,count)


def test_witnesses_cross_field_transition_dependency_without_causality_claim():
    result,training=fitted()
    edges=result["ground"]["dependency_witnesses"]
    assert any(
        edge["action"]=="tick" and
        edge["input_span"]==0 and edge["output_span"]==1
        for edge in edges)
    assert all(edge["kind"]=="witnessed_conditional_sensitivity_not_causal_proof"
               for edge in edges)


def test_partial_observation_recommends_which_unknown_bytes_to_read():
    navigator=MultiFieldNavigator(fitted()[0]["ground"])
    sample=frame(1,2)
    partial=navigator.observe_partial(sample,[(0,24),(73,74)])
    assert partial["status"]=="MULTIPLE_COMPATIBLE_MODEL_STATES"
    assert len(partial["compatible_model_states"])==4
    assert partial["next_probe"]["byte_interval"]==[311,313]
    assert partial["next_probe"]["expected_information_bits"]==2.0
    complete=navigator.observe_partial(sample,[(73,74),(311,313)])
    assert complete["status"]=="UNIQUE_COMPATIBLE_MODEL_STATE"
    assert navigator.materialize(complete["state_id"])==sample
    assert complete["scope"]=="observed_positions_only_unobserved_bytes_unverified"


def test_partial_observation_does_not_fabricate_unknown_state_bytes():
    navigator=MultiFieldNavigator(fitted()[0]["ground"])
    raw=bytearray(frame(1,2))
    raw[10]^=1  # an exact static position whose value has changed
    unknown=navigator.observe_partial(bytes(raw),[(73,74)])
    assert len(unknown["compatible_model_states"])==4
    assert unknown["status"]=="MULTIPLE_COMPATIBLE_MODEL_STATES"
    checked=navigator.observe_partial(bytes(raw),[(10,11)])
    assert checked["status"]=="CONTRADICTED"
    assert checked["compatible_model_states"]==[]


def test_navigates_a_trillion_repeated_known_actions_with_cycle_skip():
    result,training=fitted()
    nav=MultiFieldNavigator(result["ground"])
    first=bytes([0,0,255]).hex()
    horizon=10**12+3
    jump=nav.navigate(first,"tick",horizon,"bytes.snapshot",include_data=True)
    assert jump["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert jump["cycle_skipped_steps"]>10**12-20
    assert jump["evaluated_transitions"]<=8
    expected=frame(0,horizon%4)
    assert base64.b64decode(jump["result"]["data_b64"])==expected
    assert jump["result"]["sha256"]==hashlib.sha256(expected).hexdigest()
    scalar=nav.navigate(first,"toggle",10**12+1,"state.token")
    assert scalar["result"]["hex"]==bytes([1,0,255]).hex()
    digest=nav.navigate(first,"tick",10**12,"digest.sha256")
    assert digest["result"]["value"]==hashlib.sha256(frame(0,0)).hexdigest()


def test_missing_action_remains_explicit_unknown_even_after_large_horizon():
    omit=((1,3,"tick"),)
    trained,observations=fitted(omit)
    assert trained["measurement"]["unobserved_action_edges"]==1
    nav=MultiFieldNavigator(trained["ground"])
    begin=bytes([1,3,252]).hex()
    blocked=nav.navigate(begin,"tick",10**12,"digest.sha256")
    assert blocked["status"]=="UNRESOLVED"
    assert blocked["reason"]=="UNOBSERVED_STATE_ACTION"
    assert blocked["completed_steps"]==0
    assert blocked["missing"]["action"]=="tick"
    assert nav.navigate(begin,"execute shell",1)["status"]=="UNRESOLVED"
    assert nav.navigate(begin,"tick",-1)["status"]=="REFUSED"


def test_heldout_unknown_is_unresolved_not_invented():
    train,_=fixtures(((1,3,"tick"),))
    m,c=independent_step(1,3,"tick")
    hold=[(frame(1,3),"tick",frame(m,c))]
    discovered=discover_multifield_ground(train,hold)
    assert discovered["measurement"]["unresolved_heldout_count"]==1
    assert discovered["measurement"]["verified_heldout_count"]==0
    assert discovered["ground"]["heldout_unresolved"][0][
        "reason"]=="UNOBSERVED_TRANSITION"


def test_heldout_counterexample_and_training_conflict_are_refused():
    train,_=fixtures()
    dishonest=list(train)
    dishonest.append((frame(1,0),"tick",frame(1,1)))
    with pytest.raises(ValueError,match="CONTRADICTED_OBSERVED_ACTION"):
        discover_multifield_ground(dishonest)
    with pytest.raises(ValueError,match="HELDOUT_TRANSITION_CONTRADICTS"):
        discover_multifield_ground(train,[(frame(1,0),"tick",frame(1,1))])


def test_independent_challenge_preserves_contradiction_or_unknown_without_repair():
    ground,training=fitted()
    nav=MultiFieldNavigator(ground["ground"])
    good=nav.challenge([training[0],training[1]])
    assert good["status"]=="CONSISTENT_ON_CHALLENGED_OBSERVATIONS"
    before,action,after=training[0]
    changed=bytearray(after)
    changed[20]^=1
    bad=nav.challenge([(before,action,bytes(changed))])
    assert bad["status"]=="CONTRADICTED"
    assert bad["mismatches"][0]["reason"]=="STATIC_TEMPLATE_CHANGED"
    assert bad["observation_evidence_digest"]!=good["observation_evidence_digest"]
    unknown=nav.challenge([(before,"never_seen",after)])
    assert unknown["status"]=="INDETERMINATE"
    assert unknown["unknown"][0]["reason"]=="UNKNOWN_TRANSITION"
    assert nav.ground["ground_digest"]==ground["ground"]["ground_digest"]


def test_ground_integrity_and_structural_admission():
    result,observations=fitted()
    raw=result["ground"]
    changed=dict(raw)
    changed["frame_length"]=999
    with pytest.raises(ValueError,match="digest"):
        MultiFieldNavigator(changed)
    changed=dict(raw)
    changed["variable_spans"]=[{"offset":73,"width":300}]
    changed["ground_digest"]=domain_digest("vl64.multifield-action-ground.v1",
            {k:v for k,v in changed.items() if k!="ground_digest"})
    with pytest.raises(ValueError):
        MultiFieldNavigator(changed)


def test_invalid_partial_intervals_and_bad_state_selection_are_refused():
    nav=MultiFieldNavigator(fitted()[0]["ground"])
    sample=frame(0,0)
    with pytest.raises(ValueError):
        nav.observe_partial(sample,[(12,5)])
    with pytest.raises(ValueError):
        nav.observe_partial(sample,[(73,74),(72,73)])
    with pytest.raises(ValueError):
        nav.observe_partial(sample,[])
    assert nav.navigate("ffff","tick",1)["status"]=="UNRESOLVED"
    assert nav.navigate("nothex","tick",1)["status"]=="REFUSED"



def test_mixed_action_programs_reconstruct_independent_reference_states():
    import itertools
    nav=MultiFieldNavigator(fitted()[0]["ground"])
    initial=bytes([0,0,255]).hex()
    tested=0
    for size in range(1,6):
        for word in itertools.product(("tick","toggle"),repeat=size):
            model=nav.navigate_word(initial,word,"digest.sha256")
            assert model["status"]=="OBSERVED_GROUND_NAVIGATION"
            assert len(model["witnessed_edge_receipts"])==len(word)
            mode,count=0,0
            for action in word:
                mode,count=independent_step(mode,count,action)
            assert model["result"]["value"]==hashlib.sha256(
                frame(mode,count)).hexdigest()
            tested+=1
    assert tested==62


def test_mixed_action_program_does_not_complete_missing_edge():
    nav=MultiFieldNavigator(fitted(((1,3,"tick"),))[0]["ground"])
    blocked=nav.navigate_word(bytes([1,3,252]).hex(),["tick","toggle"])
    assert blocked["status"]=="UNRESOLVED"
    assert blocked["completed_steps"]==0
    assert blocked["missing"]["action"]=="tick"
    assert nav.navigate_word("00ff",[])["status"]=="REFUSED"
