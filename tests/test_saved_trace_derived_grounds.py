"""Archived observations really reproduce active planning and modeled navigation."""
import json
from pathlib import Path

from vld import ObserverIntention, domain_digest
from experiments.byte_fabric.trace_synthesis import TraceLedger,TraceGroundSynthesizer
from experiments.byte_fabric.active_ground import ActionHypothesis,ActionGroundNavigator
from experiments.observer_intent.arbitrate import arbitrate,render_result


ROOT=Path(__file__).resolve().parents[1]
BASE="experiments/byte_fabric/datasets/"


def saved(name):
    return json.loads((ROOT/(BASE+name)).read_text(encoding="utf-8"))


def test_saved_initial_traces_regenerate_competing_ground_models():
    original=saved("2026-11-18-initial-trace-ledger.json")
    trace=TraceLedger.from_snapshot(original)
    assert len(trace.traces())==2
    synthesis=TraceGroundSynthesizer(max_states=2).discover(trace)
    assert synthesis["candidate_count"]==14
    assert synthesis["completeness_claim"] is False
    assert all(meta["unseen_transitions_certified"] is False
               for meta in synthesis["candidate_metadata"])
    assert original["ledger_digest"]==domain_digest(
        "vl64.action-trace-ledger.v1",
        {k:v for k,v in original.items() if k!="ledger_digest"})


def test_saved_final_trace_history_reproduces_ground_without_hidden_transition_code():
    original=saved("2026-11-18-observed-trace-ledger.json")
    observations=TraceLedger.from_snapshot(original)
    ground=saved("2026-11-18-trace-derived-action-ground.json")
    result=TraceGroundSynthesizer(max_states=2).discover(observations)
    assert result["status"]=="SYNTHESIZED_BOUNDED_CANDIDATES"
    assert ground["ground_digest"] in {
        model["ground_digest"] for model in result["candidates"]}
    assert len(ground["states"])==2
    for word,outputs in observations.traces():
        assert ActionHypothesis(ground).predict(word)==outputs
    assert ActionGroundNavigator(ground).repeat("flip",10**12)[
        "terminal_state"]==0


def test_saved_intentions_are_callable_through_existing_observer_cli():
    for filename,status in (
        ("trace_derived_next_probe.json","QUALIFIED_INFORMATIVE_PROBE"),
        ("trace_derived_trillion.json","MODELED_ACTION_NAVIGATION"),
    ):
        path=ROOT/"experiments/byte_fabric/intents"/filename
        req=ObserverIntention.from_dict(json.loads(path.read_text()))
        receipt=arbitrate(req,byte_root=ROOT)
        assert receipt["status"]==status
        assert receipt["plan"]["execution_authority"]==(
            "no_host_actions_performed_by_this_observer_route")
        assert json.loads(render_result(receipt,"json"))["status"]==status


def test_recorded_live_bytes_are_actual_pinned_public_sources():
    result=saved("2026-11-18-trace-derived-measurements.json")
    report=result["controlled"]
    assert report["initial_candidate_count"]==14
    assert report["active_discriminating_rounds"]==3
    assert report["heldout_query_words_checked"]==62
    assert report["heldout_mismatches"]==[]
    assert report["selected_hypothetical_edges"]==0
    assert report["reference_returned_output_bytes"]==68
    assert result["cross_runtime"]["selected_ground_digest_identical"]
    assert result["cross_runtime"]["initial_ledger_digest_identical"]
    assert result["cross_runtime"]["compact_core_bytes"]<=65536
    external=result["real_network"]
    assert external["status"]=="OBSERVED_REAL_PUBLIC_NETWORK_BYTES"
    assert len(external["read_evidence"])>=2
    assert all(x["http_status"]==200 and x["bytes_received"]>1000 and
               x["transport_kind"]=="https" and
               len(x["sha256"])==64
               for x in external["read_evidence"])
    assert "not_remote_state_machine" in external["scope"]
