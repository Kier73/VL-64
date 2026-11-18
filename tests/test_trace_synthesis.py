"""Actual observations synthesize the hypotheses; no supplied model catalogue."""
from __future__ import annotations

import itertools
import json

import pytest

from vld import domain_digest
from experiments.byte_fabric.active_ground import (
    ActiveGroundSession,ActionGroundNavigator,ActionHypothesis,
    QualifiedResetProbe,
)
from experiments.byte_fabric.trace_synthesis import (
    TraceLedger,TraceGroundSynthesizer,TraceDrivenDiscovery,
)


ACTIONS=("flip","read")


def independent_oracle():
    """Host reference implementation intentionally has no ActionGround object."""
    def run(word):
        state=0
        observations=[]
        for action in word:
            if action=="flip":
                state^=1
                observations.append(b"ok")
            elif action=="read":
                observations.append(b"one" if state else b"zero")
            else:
                raise ValueError("Invalid action")
        return tuple(observations)
    return QualifiedResetProbe(ACTIONS,run,source="independent-two-state-reference")


def boot():
    host=independent_oracle()
    ledger=TraceLedger(ACTIONS)
    ledger.query(host,("read",))
    ledger.query(host,("flip",))
    return ledger,host


def test_trace_tree_is_built_only_from_observed_bytes_and_unknowns_stay_guesses():
    ledger,host=boot()
    found=TraceGroundSynthesizer(max_states=2,max_candidates=24).discover(ledger)
    assert found["status"]=="SYNTHESIZED_BOUNDED_CANDIDATES"
    assert found["candidate_count"]>=2
    assert found["observed_action_edges"]==2
    assert found["observed_prefix_tree_nodes"]==3
    assert found["observed_output_symbols_hex"]==[b"ok".hex(),b"zero".hex()]
    assert found["completeness_claim"] is False
    assert all(x["unseen_transitions_certified"] is False
               for x in found["candidate_metadata"])
    assert any(x["hypothesized_state_action_edges"]>0
               for x in found["candidate_metadata"])
    for ground in found["candidates"]:
        model=ActionHypothesis(ground)
        assert model.predict(("read",))==(b"zero",)
        assert model.predict(("flip",))==(b"ok",)
    assert found["synthesis_digest"]==domain_digest(
        "vl64.trace-to-ground-synthesis.v1",
        {k:v for k,v in found.items() if k!="synthesis_digest"})


def test_active_probe_uses_generated_candidates_and_resynthesizes_after_new_output():
    ledger,host=boot()
    first=TraceGroundSynthesizer(max_states=2,max_candidates=24).discover(ledger)
    controller=TraceDrivenDiscovery(host,max_states=2,max_candidates=24)
    controller.ledger=ledger
    step=controller.step(max_depth=3,max_words=64)
    assert step["status"]=="RESYNTHESIZED_FROM_NEW_EVIDENCE"
    assert step["selected_plan"]["expected_information_bits"]>0
    assert step["observed_action_evidence"]["status"]=="OBSERVED_ACTION_TRACE"
    assert step["updated_synthesis_digest"]!=first["synthesis_digest"]
    assert controller.ledger.snapshot()["observations"][-1]["source"]==(
        "independent-two-state-reference")
    latest=controller.synthesizer.discover(controller.ledger)
    for ground in latest["candidates"]:
        model=ActionHypothesis(ground)
        for word,outputs in controller.ledger.traces():
            assert model.predict(word)==outputs


def test_counterexample_new_output_revises_guessed_output_alphabet():
    ledger,host=boot()
    before=TraceGroundSynthesizer(max_states=2).discover(ledger)
    assert b"one".hex() not in before["observed_output_symbols_hex"]
    record=ledger.query(host,("flip","read"))
    assert record["status"]=="OBSERVED_ACTION_TRACE"
    after=TraceGroundSynthesizer(max_states=2).discover(ledger)
    assert b"one".hex() in after["observed_output_symbols_hex"]
    assert after["synthesis_digest"]!=before["synthesis_digest"]
    assert after["candidate_count"]>=1
    assert all(ActionHypothesis(g).predict(("flip","read"))==(
        b"ok",b"one") for g in after["candidates"])


def test_observed_reset_inconsistency_is_retained_and_model_synthesis_refuses():
    ledger,host=boot()
    existing=ledger.snapshot()["ledger_digest"]
    record=ledger.observe(("read",),(b"unexpected",),source="repeat-test")
    assert record["status"]=="CONTRADICTED_REPEATED_OBSERVATION"
    assert record["conflicts"][0]["previous_sha256"]!=record["conflicts"][0]["new_sha256"]
    assert ledger.snapshot()["ledger_digest"]!=existing
    fail=TraceGroundSynthesizer().discover(ledger)
    assert fail["status"]=="CONTRADICTED_RESET_DETERMINISM"
    assert fail["candidate_count"] if "candidate_count" in fail else True
    assert fail["candidates"]==[]


def test_with_sufficient_observed_actions_model_predicts_heldout_traces():
    ledger,host=boot()
    for word in (
        ("flip","read"),
        ("flip","flip","read"),
        ("flip","flip"),
        ("read","flip","read"),
        ("flip","read","flip","read"),
        ("flip","flip","flip","read"),
    ):
        ledger.query(host,word)
    trained=TraceGroundSynthesizer(max_states=2,max_candidates=24,
                                   max_nodes=30,max_assignments=50000).discover(ledger)
    assert trained["status"]=="SYNTHESIZED_BOUNDED_CANDIDATES"
    assert trained["candidate_count"]>=1
    # Every fully observed state/action cell in this example constrains the
    # 2-state graph. An independent holdout is not fed into the synthesizer.
    holdout=[]
    for length in range(1,6):
        for word in itertools.product(ACTIONS,repeat=length):
            holdout.append((word,host.run(word)))
    assert any(all(ActionHypothesis(g).predict(word)==outputs
                   for word,outputs in holdout)
               for g in trained["candidates"])
    passing=[g for g in trained["candidates"] if all(
        ActionHypothesis(g).predict(word)==outputs for word,outputs in holdout)]
    jump=ActionGroundNavigator(passing[0]).repeat("flip",10**12)
    assert jump["status"]=="MODELED_ACTION_NAVIGATION"
    assert jump["terminal_state"]==0
    assert jump["last_output_hex"]==b"ok".hex()
    assert jump["power_compositions"]<100


def test_traces_with_empty_output_and_prefix_conflicts():
    actions=("ping",)
    ledger=TraceLedger(actions)
    ledger.observe(("ping",),(b"",))
    assert ledger.snapshot()["contradiction_count"]==0
    ledger.observe(("ping",),(b"changed",))
    assert ledger.snapshot()["contradiction_count"]==1
    assert TraceGroundSynthesizer().discover(ledger)["status"]==(
        "CONTRADICTED_RESET_DETERMINISM")


def test_missing_data_and_budget_refusal_behaviors():
    ledger=TraceLedger(ACTIONS)
    assert TraceGroundSynthesizer().discover(ledger)["status"]=="UNRESOLVED"
    with pytest.raises(ValueError):
        ledger.observe(("execute shell",),(b"",))
    with pytest.raises(ValueError):
        ledger.observe(("flip",),(b"A",b"B"))
    with pytest.raises(ValueError):
        TraceGroundSynthesizer(max_states=0)
    with pytest.raises(ValueError):
        TraceGroundSynthesizer(max_candidates=100)
    with pytest.raises(ValueError):
        TraceGroundSynthesizer(max_assignments=1)
    ledger,host=boot()
    exhausted=TraceGroundSynthesizer(
        max_states=2,max_nodes=2).discover(ledger)
    assert exhausted["status"]=="UNRESOLVED"
    assert exhausted["reason"]=="OBSERVATION_TREE_BUDGET"
    with pytest.raises(ValueError):
        ledger.query(QualifiedResetProbe(("other",),lambda w:(b"",)),("read",))


def test_version_space_never_claims_global_uniqueness():
    ledger,host=boot()
    found=TraceGroundSynthesizer(max_states=1).discover(ledger)
    assert found["status"]=="SYNTHESIZED_BOUNDED_CANDIDATES"
    assert found["completeness_claim"] is False
    assert "unknown_edge_guesses" in found["scope"]
    # If a previously unknown output appears, all previous candidates can be
    # contradicted, but fresh synthesis must keep the actual new evidence.
    model=ActionHypothesis(found["candidates"][0])
    assert model.predict(("flip","read")) != host.run(("flip","read"))
    ledger.query(host,("flip","read"))
    revised=TraceGroundSynthesizer(max_states=2).discover(ledger)
    assert revised["observed_action_edges"]>found["observed_action_edges"]



def test_ledger_snapshot_remains_identical_after_new_observations():
    ledger,host=boot()
    snapshot=ledger.snapshot()
    old_digest=snapshot["ledger_digest"]
    ledger.query(host,("flip","read"))
    assert snapshot["ledger_digest"]==old_digest
    assert len(snapshot["observations"])==2
    assert len(ledger.snapshot()["observations"])==3
    assert TraceLedger.from_snapshot(snapshot).snapshot()==snapshot
