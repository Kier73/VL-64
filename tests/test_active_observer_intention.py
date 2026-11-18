"""Saved active experiment descriptions through the external observer channel."""
import json

from vld import ObserverIntention
from experiments.byte_fabric.active_lab import (
    controlled_experiment, candidate_models, independent_host,
)
from experiments.byte_fabric.active_adapter import (
    arbitrate_active,render_active_receipt,
)
from experiments.byte_fabric.active_ground import (
    ActiveGroundSession,ActionGroundNavigator,
)
from experiments.observer_intent.arbitrate import arbitrate,render_result


def test_saved_candidate_catalogue_recommends_informative_action(tmp_path):
    result=controlled_experiment()
    path=tmp_path/"models.json"
    path.write_text(json.dumps(result["candidates"]))
    request=ObserverIntention(
        "ground.active.plan", "workspace:models.json", output="json",
        options={"max_depth":2,"max_words":32})
    receipt=arbitrate(request,byte_root=tmp_path)
    assert receipt["status"]=="QUALIFIED_INFORMATIVE_PROBE"
    assert len(receipt["result"]["word"])==2
    assert receipt["result"]["expected_information_bits"]==1.0
    assert receipt["plan"]["execution_authority"]==(
        "no_host_actions_performed_by_this_observer_route")
    assert json.loads(render_result(receipt,"json"))["status"]==receipt["status"]
    assert len(render_result(receipt,"jsonl").splitlines())==1
    assert "Recommended actions:" in render_result(receipt,"text")


def test_observer_pure_action_navigation_never_executes_host_action(tmp_path):
    result=controlled_experiment()
    path=tmp_path/"selected.json"
    path.write_text(json.dumps(result["selected_ground"]))
    request=ObserverIntention(
        "ground.action.navigate","workspace:selected.json",output="json",
        options={"action":"advance","steps":10**12})
    receipt=arbitrate(request,byte_root=tmp_path)
    assert receipt["status"]=="MODELED_ACTION_NAVIGATION"
    assert receipt["result"]["terminal_state"]==1
    assert receipt["result"]["steps"]==10**12
    assert receipt["result"]["power_compositions"]<100
    assert receipt["verification_scope"]=="description_only_not_an_executed_remote_probe"
    assert json.loads(render_active_receipt(receipt,"json"))["status"]==receipt["status"]


def test_action_intention_rejects_malformed_and_nonexistent_ground(tmp_path):
    request=ObserverIntention(
        "ground.action.navigate","workspace:missing.json",
        options={"steps":10**12,"action":"advance"})
    assert arbitrate(request,byte_root=tmp_path)["status"]=="UNRESOLVED"
    result=controlled_experiment()
    path=tmp_path/"model.json"
    path.write_text(json.dumps(result["selected_ground"]))
    invalid=ObserverIntention(
        "ground.action.navigate","workspace:model.json",
        options={"steps":"many","action":"advance"})
    assert arbitrate(invalid,byte_root=tmp_path)["status"]=="REFUSED"
    invalid=ObserverIntention(
        "ground.action.navigate","workspace:model.json",
        options={"steps":10,"action":"unknown"})
    assert arbitrate(invalid,byte_root=tmp_path)["status"]=="UNRESOLVED"


def test_proposed_plan_cannot_forge_predicted_outputs_with_rehashed_digest():
    from vld import domain_digest
    session=ActiveGroundSession(candidate_models())
    p=session.plan(max_depth=2)
    p["predicted_partitions"][0]["outputs_hex"]=["never","seen"]
    p["plan_digest"]=domain_digest("vl64.active-ground-probe-plan.v1",{
        k:v for k,v in p.items() if k!="plan_digest"})
    observed=independent_host(2).run(p["word"])
    import pytest
    with pytest.raises(ValueError,match="predictions"):
        session.apply(p,observed)
    assert len(session.evidence)==0
