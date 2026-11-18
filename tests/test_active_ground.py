"""Action-conditioned candidate Grounds, discriminating experiments, refusals."""
from __future__ import annotations

import hashlib

import pytest

from vld import domain_digest
from experiments.byte_fabric.active_ground import (
    ActionGroundNavigator, ActionHypothesis, ActiveGroundSession,
    QualifiedResetProbe, action_ground,
)


ALPHABET=("advance","inspect","switch")


def candidates():
    """Four models agree initially but disagree on two future experiments."""
    records=[]
    for i in range(4):
        first=b"x" if i in (0,1) else b"y"
        second=b"left" if i in (0,2) else b"right"
        states=[
            {
                "advance":{"next":1,"output_hex":b"tick".hex()},
                "inspect":{"next":0,"output_hex":b"ready".hex()},
                "switch":{"next":2,"output_hex":b"tick".hex()},
            },
            {
                "advance":{"next":1,"output_hex":b"tick".hex()},
                "inspect":{"next":1,"output_hex":first.hex()},
                "switch":{"next":1,"output_hex":b"tick".hex()},
            },
            {
                "advance":{"next":2,"output_hex":b"tick".hex()},
                "inspect":{"next":2,"output_hex":second.hex()},
                "switch":{"next":2,"output_hex":b"tick".hex()},
            },
        ]
        records.append(action_ground(ALPHABET,states))
    return records


def independent_oracle(variant):
    """Not derived from model descriptor: independent resettable reference."""
    def run(word):
        state=0
        outputs=[]
        for action in word:
            if state==0:
                if action=="inspect":
                    output=b"ready"
                else:
                    state=1 if action=="advance" else 2
                    output=b"tick"
            elif state==1:
                output=(b"x" if variant in (0,1) else b"y") if action=="inspect" else b"tick"
            else:
                output=(b"left" if variant in (0,2) else b"right") if action=="inspect" else b"tick"
            outputs.append(output)
        return tuple(outputs)
    return QualifiedResetProbe(ALPHABET,run,policy="local-simulated",
                               source="independent-controlled-oracle")


def test_active_plan_finds_informative_two_step_action_word():
    session=ActiveGroundSession(candidates())
    plan=session.plan(max_depth=2,max_words=20)
    assert plan["status"]=="QUALIFIED_INFORMATIVE_PROBE"
    assert len(plan["word"])==2
    assert plan["word"][1]=="inspect"
    assert plan["word"][0] in ("advance","switch")
    assert plan["expected_information_bits"]==1.0
    assert len(plan["predicted_partitions"])==2
    assert plan["candidate_count"]==4
    assert plan["declared_cost"]==2
    assert plan["score_bits_per_declared_cost"]==0.5


def test_replan_after_first_observation_eliminates_remaining_hypothesis():
    session=ActiveGroundSession(candidates())
    oracle=independent_oracle(variant=2)
    records=[]
    for _ in range(2):
        result=session.step(oracle,max_depth=2,max_words=30)
        assert result["status"] in ("HYPOTHESES_REFINED","ONE_HYPOTHESIS_REMAINS")
        records.append(result)
    assert records[0]["status"]=="HYPOTHESES_REFINED"
    assert len(records[0]["observation"]["after_hypothesis_digests"])==2
    assert records[1]["status"]=="ONE_HYPOTHESIS_REMAINS"
    assert records[0]["plan"]["word"]!=records[1]["plan"]["word"]
    assert len(session.models)==1
    assert session.models[0].digest==candidates()[2]["ground_digest"]
    result=session.receipt()
    assert result["status"]=="ONE_HYPOTHESIS_REMAINS"
    assert len(result["observations"])==2
    assert result["remaining_ground_digests"]==[candidates()[2]["ground_digest"]]


def test_observation_contradicts_all_models_without_rewriting_evidence():
    session=ActiveGroundSession(candidates())
    hidden=independent_oracle(variant=0)
    def contradict(_word):
        return (b"unknown",b"contradiction")
    host=QualifiedResetProbe(ALPHABET,contradict)
    result=session.step(host,max_depth=2,max_words=30)
    assert result["status"]=="CONTRADICTED_ALL_HYPOTHESES"
    assert len(session.models)==0
    assert result["observation"]["after_hypothesis_digests"]==[]
    assert len(result["observation"]["eliminated_hypothesis_digests"])==4
    assert result["observation"]["observed_output_digests"][0]==hashlib.sha256(
        b"unknown").hexdigest()
    assert session.step(host)["status"]=="CONTRADICTED_ALL_HYPOTHESES"


def test_probe_plan_cannot_be_replayed_on_a_stale_version_space():
    session=ActiveGroundSession(candidates())
    p=session.plan(max_depth=2)
    actual=independent_oracle(0).run(p["word"])
    first=session.apply(p,actual,probe_source="test")
    assert first["status"]=="HYPOTHESES_REFINED"
    with pytest.raises(ValueError,match="Stale"):
        session.apply(p,actual)
    forged=dict(session.plan(max_depth=2))
    forged["expected_information_bits"]=200.0
    with pytest.raises(ValueError,match="digest"):
        session.apply(forged,actual)


def test_incomplete_or_unauthorized_reset_oracle_never_refines_ground():
    session=ActiveGroundSession(candidates())
    def incomplete(word):
        return (b"x",)
    bad=QualifiedResetProbe(ALPHABET,incomplete)
    response=session.step(bad,max_depth=2)
    assert response["status"]=="UNRESOLVED"
    assert response["reason"]=="PROBE_FAILED"
    assert len(session.models)==4
    assert len(session.evidence)==0
    with pytest.raises(ValueError):
        QualifiedResetProbe(ALPHABET,lambda _:(),policy="write-unapproved")
    with pytest.raises(ValueError):
        QualifiedResetProbe(("unqualified action",),lambda _:())


def test_bounded_search_does_not_conclude_hypotheses_are_equivalent():
    session=ActiveGroundSession(candidates())
    limited=session.plan(max_depth=1,max_words=3)
    assert limited["status"]=="UNRESOLVED"
    assert limited["reason"]=="NO_DISCRIMINATOR_WITHIN_SEARCH_BUDGET"
    assert len(session.models)==4
    with pytest.raises(ValueError):
        session.plan(max_depth=100)
    with pytest.raises(ValueError):
        session.plan(max_words=100000)


def test_action_ground_schema_and_untrusted_output_are_strict():
    raw=candidates()[0]
    assert ActionHypothesis(raw).predict(("advance","inspect"))==(b"tick",b"x")
    modified=dict(raw)
    modified["initial_state"]=2
    with pytest.raises(ValueError,match="digest"):
        ActionHypothesis(modified)
    invalid_states=[dict(x) for x in raw["states"]]
    invalid_states[0]={k:dict(v) for k,v in invalid_states[0].items()}
    invalid_states[0]["advance"]["next"]=55
    with pytest.raises(ValueError):
        action_ground(ALPHABET,invalid_states)
    invalid_states[0]["advance"]["next"]=1
    invalid_states[0]["advance"]["output_hex"]="non-hex-code"
    with pytest.raises(ValueError):
        action_ground(ALPHABET,invalid_states)


def test_modeled_repeated_action_is_powered_not_replayed():
    ground=candidates()[2]
    nav=ActionGroundNavigator(ground)
    result=nav.repeat("advance",10**12)
    assert result["status"]=="MODELED_ACTION_NAVIGATION"
    assert result["steps"]==10**12
    assert result["terminal_state"]==1
    assert result["last_output_hex"]==b"tick".hex()
    assert result["last_output_sha256"]==hashlib.sha256(b"tick").hexdigest()
    assert result["power_compositions"]<100
    assert result["scope"]=="hypothetical-deterministic-model-not-remote-action-execution"
    for n in (0,1,2,3,11,1000):
        actual=ActionHypothesis(ground).predict(("advance",)*n) if n<=11 else (b"tick",)
        expected=b"tick" if n else None
        check=nav.repeat("advance",n)
        assert check["last_output_hex"]==(expected.hex() if expected else None)
    assert nav.repeat("unapproved",3)["status"]=="UNRESOLVED"
    assert nav.repeat("advance",-1)["status"]=="REFUSED"


def test_action_ground_requires_canonical_constrained_model():
    with pytest.raises(ValueError):
        action_ground(["a","a"],[{"a":{"next":0,"output_hex":""}}])
    with pytest.raises(ValueError):
        action_ground(["a"],[{"a":{"next":0,"output_hex":"AA"}}])
    with pytest.raises(ValueError):
        action_ground(["a"],[])
    session=ActiveGroundSession(candidates())
    with pytest.raises(ValueError):
        session.step(QualifiedResetProbe(["inspect"],lambda word:(b"a",)))
    with pytest.raises(ValueError):
        ActiveGroundSession(candidates(),costs={"advance":0})
