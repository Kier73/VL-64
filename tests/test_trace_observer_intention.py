"""External observer can synthesize Grounds directly from recorded action bytes."""
import json

import pytest
from vld import ObserverIntention, domain_digest
from experiments.byte_fabric.trace_synthesis import (
    TraceLedger,TraceGroundSynthesizer,
)
from experiments.byte_fabric.active_ground import QualifiedResetProbe
from experiments.observer_intent.arbitrate import arbitrate,render_result


def evidence():
    ledger=TraceLedger(("flip","read"))
    ledger.observe(("read",),(b"zero",),source="independent-oracle",
                   policy="local-simulated")
    ledger.observe(("flip",),(b"ok",),source="independent-oracle",
                   policy="local-simulated")
    return ledger


def test_observer_returns_candidate_grounds_from_raw_trace_evidence(tmp_path):
    observed=evidence()
    path=tmp_path/"traces.json"
    path.write_text(json.dumps(observed.snapshot()))
    req=ObserverIntention(
        "ground.traces.plan","workspace:traces.json",output="json",
        options={"max_states":2,"max_depth":3,"max_words":64})
    receipt=arbitrate(req,byte_root=tmp_path)
    assert receipt["status"]=="QUALIFIED_INFORMATIVE_PROBE"
    assert receipt["result"]["synthesis"]["candidate_count"]>=2
    assert receipt["result"]["synthesis"]["completeness_claim"] is False
    assert receipt["result"]["recommendation"]["word"]
    assert receipt["plan"]["execution_authority"]==(
        "no_host_actions_performed_by_this_observer_route")
    assert len(receipt["result"]["candidate_ground_digests"])>=2
    assert "Recommended actions:" in render_result(receipt,"text")
    assert json.loads(render_result(receipt,"json"))["status"]==receipt["status"]


def test_evidence_ledger_roundtrip_checks_every_record_and_report(tmp_path):
    snapshot=evidence().snapshot()
    assert TraceLedger.from_snapshot(snapshot).snapshot()==snapshot
    forged=json.loads(json.dumps(snapshot))
    forged["observations"][0]["outputs_hex"][0]="66616b65"
    forged["ledger_digest"]=domain_digest("vl64.action-trace-ledger.v1",{
        k:v for k,v in forged.items() if k!="ledger_digest"})
    with pytest.raises(ValueError,match="receipt"):
        TraceLedger.from_snapshot(forged)
    forged=json.loads(json.dumps(snapshot))
    forged["actions"].reverse()
    with pytest.raises(ValueError,match="digest"):
        TraceLedger.from_snapshot(forged)


def test_trace_intention_does_not_fetch_for_unadmitted_input(tmp_path):
    snapshot=evidence().snapshot()
    (tmp_path/"traces.json").write_text(json.dumps(snapshot))
    req=ObserverIntention("ground.traces.plan","workspace:traces.json",
        options={"install_operator":"arbitrary"})
    result=arbitrate(req,byte_root=tmp_path)
    assert result["status"]=="REFUSED"
    assert result["reason"]=="INVALID_TRACE_SYNTHESIS_OPTIONS"


def test_trace_intention_refuses_unqualified_or_contradicted_dataset(tmp_path):
    ledger=evidence()
    ledger.observe(("read",),(b"other",))
    (tmp_path/"traces.json").write_text(json.dumps(ledger.snapshot()))
    req=ObserverIntention(
        "ground.traces.plan","workspace:traces.json",
        options={"max_states":2})
    result=arbitrate(req,byte_root=tmp_path)
    assert result["status"]=="UNRESOLVED"
    assert result["result"]["synthesis"]["status"]==(
        "CONTRADICTED_RESET_DETERMINISM")
    assert result["result"]["recommendation"]["status"]=="UNRESOLVED"


def test_trace_intention_refuses_duplicate_json_keys(tmp_path):
    (tmp_path/"traces.json").write_text('{"schema":"first","schema":"second"}')
    req=ObserverIntention(
        "ground.traces.plan","workspace:traces.json",
        options={"max_states":2})
    receipt=arbitrate(req,byte_root=tmp_path)
    assert receipt["status"]=="UNRESOLVED"
