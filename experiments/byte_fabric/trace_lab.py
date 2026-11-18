"""'Use the system': induce, challenge and navigate action Grounds from bytes.

The hidden resettable reference machine has no access to the synthesizer's
candidate model catalogue. The learner sees only authorized input/output
traces. No active probe authorizes an arbitrary remote mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

from vld import ObserverIntention
from .active_ground import ActionGroundNavigator, ActionHypothesis, QualifiedResetProbe
from .trace_synthesis import TraceDrivenDiscovery,TraceGroundSynthesizer
from .sources import acquire


ALPHABET=("flip","read")
PINNED_REVISION="d1291a08a3f2b073da6e5b2eef1f8e2211e74a36"


def controlled_oracle(metrics=None):
    """Actual transition implementation, not a supplied ActionHypothesis."""
    def run(word):
        state=0
        observed=[]
        for action in word:
            if action=="flip":
                state=1-state
                observed.append(b"ok")
            elif action=="read":
                observed.append(b"one" if state else b"zero")
            else:
                raise ValueError("Action not admitted")
        if metrics is not None:
            metrics["requests"]+=1
            metrics["action_steps"]+=len(word)
            metrics["output_bytes"]+=sum(len(x) for x in observed)
        return tuple(observed)
    return QualifiedResetProbe(ALPHABET,run,source="independent-controlled-machine",
                               policy="local-simulated")


def controlled_experiment():
    metrics={"requests":0,"action_steps":0,"output_bytes":0}
    host=controlled_oracle(metrics)
    research=TraceDrivenDiscovery(host,max_states=2,max_candidates=24)
    start=time.perf_counter_ns()
    initial=research.seed((("read",),("flip",)))
    initial_ledger=research.ledger.snapshot()
    if initial["candidate_count"]<2:
        raise RuntimeError("Initial observations did not admit multiple hypotheses")
    early=[]
    for _ in range(3):
        result=research.step(max_depth=3,max_words=64)
        early.append(result)
        if result["status"]!="RESYNTHESIZED_FROM_NEW_EVIDENCE":
            break
    initial_alphabet=set(initial["observed_output_symbols_hex"])
    # Directly request relevant access/suffix coverage not guaranteed by the
    # bounded information-gain search. Those are *training* traces, never heldout.
    coverage=(
        ("flip","read"),("flip","flip","read"),
        ("flip","flip"),("read","flip","read"),
        ("flip","read","flip","read"),
    )
    for word in coverage:
        research.ledger.query(host,word)
    complete=research.synthesizer.discover(research.ledger)
    if complete["status"]!="SYNTHESIZED_BOUNDED_CANDIDATES":
        raise RuntimeError("Cannot synthesize model from expanded evidence")
    # Select using TRAINING evidence only. No heldout word chooses the model.
    metadata={item["ground_digest"]:item for item in complete["candidate_metadata"]}
    qualified=sorted(
        complete["candidates"],
        key=lambda g: (
            metadata[g["ground_digest"]]["hypothesized_state_action_edges"],
            len(g["states"]),g["ground_digest"]),
    )
    chosen=qualified[0]
    predicted=ActionHypothesis(chosen)
    validation_oracle=controlled_oracle()
    holdout=[
        (word,validation_oracle.run(word))
        for depth in range(1,6)
        for word in itertools.product(ALPHABET,repeat=depth)
    ]
    wrong=[list(word) for word,outputs in holdout
           if predicted.predict(word)!=outputs]
    tested=len(holdout)
    if wrong:
        raise RuntimeError("Selected trace-derived Ground fails independent heldout")
    horizon=10**12
    navigate=ActionGroundNavigator(chosen).repeat("flip",horizon)
    if (navigate["status"]!="MODELED_ACTION_NAVIGATION" or
            navigate["terminal_state"]!=0 or
            navigate["last_output_hex"]!=b"ok".hex()):
        raise RuntimeError("Selected modeled transition navigation disagrees")
    final=research.receipt()
    elapsed_ms=(time.perf_counter_ns()-start)/1e6
    return {
        "initial_ledger":initial_ledger,
        "ledger":research.ledger.snapshot(),
        "ground":chosen,
        "candidate_catalogue":{
            "schema":"vl64.action-candidate-catalogue.v1",
            "candidates":complete["candidates"],
            "costs":{name:1 for name in ALPHABET},
        },
        "synthesis":complete,
        "session":final,
        "report":{
            "schema":"vl64.trace-derived-active-learning-lab.v1",
            "status":"HELDOUT_VALIDATED_TRACE_DERIVED_GROUND",
            "initial_observed_words":[["read"],["flip"]],
            "initial_candidate_count":initial["candidate_count"],
            "active_discriminating_rounds":len([
                r for r in early
                if r["status"]=="RESYNTHESIZED_FROM_NEW_EVIDENCE"]),
            "active_probes":[r.get("selected_plan",{}).get("word")
                             for r in early if r.get("selected_plan")],
            "first_observed_output_alphabet_hex":sorted(initial_alphabet),
            "final_observed_output_alphabet_hex":complete["observed_output_symbols_hex"],
            "observed_trace_count":len(research.ledger.traces()),
            "final_candidate_count":complete["candidate_count"],
            "selected_ground_digest":chosen["ground_digest"],
            "selected_state_count":len(chosen["states"]),
            "selected_hypothetical_edges":
                metadata[chosen["ground_digest"]]["hypothesized_state_action_edges"],
            "training_source":"independent-controlled-resettable-byte-machine",
            "heldout_query_words_checked":tested,
            "heldout_mismatches":wrong,
            "reference_probe_calls":metrics["requests"],
            "reference_action_steps":metrics["action_steps"],
            "reference_returned_output_bytes":metrics["output_bytes"],
            "modeled_horizon":horizon,
            "modeled_terminal_state":navigate["terminal_state"],
            "power_compositions":navigate["power_compositions"],
            "measured_discovery_ms_one_run":round(elapsed_ms,4),
            "model_scope":"bounded_two_state_deterministic_mealy_hypothesis",
            "global_equivalence_proven":False,
            "core_modified":False,
        }
    }


def real_readonly_experiment():
    """Learn candidates from actual pinned GitHub GET bytes, not model files."""
    root=Path.cwd()
    urls={
        "read_readme":f"https://raw.githubusercontent.com/Kier73/VL-64/{PINNED_REVISION}/README.md",
        "read_goals":f"https://raw.githubusercontent.com/Kier73/VL-64/{PINNED_REVISION}/docs/RESEARCH_GOALS.md",
    }
    reads=[]
    def oracle(word):
        out=[]
        for action in word:
            frame=acquire(urls[action],root=root,max_bytes=65536)
            if not frame["complete"]:
                raise ValueError("Remote source was incomplete")
            content=frame["data"]
            out.append(hashlib.sha256(content).digest())
            reads.append({
                "source":urls[action],
                "bytes_received":len(content),
                "sha256":hashlib.sha256(content).hexdigest(),
                "transport":frame["transport"],
                "content_scope":"complete_pinned_public_read_only",
            })
        return tuple(out)
    host=QualifiedResetProbe(tuple(sorted(urls)),oracle,
                             policy="read-only-remote",
                             source="pinned-public-github-raw-bytes")
    research=TraceDrivenDiscovery(host,max_states=2,max_candidates=24)
    seeded=research.seed((("read_goals",),("read_readme",)))
    if seeded["status"]!="SYNTHESIZED_BOUNDED_CANDIDATES":
        raise RuntimeError("Cannot synthesize candidates from Internet bytes")
    planned=research.step(max_depth=2,max_words=32)
    result={
        "schema":"vl64.trace-derived-remote-byte-experiment.v1",
        "status":"OBSERVED_REAL_PUBLIC_NETWORK_BYTES",
        "source_revision":PINNED_REVISION,
        "initial_candidate_count":seeded["candidate_count"],
        "initial_observed_output_symbols":seeded["observed_output_symbols_hex"],
        "active_round_status":planned["status"],
        "active_word":planned.get("selected_plan",{}).get("word"),
        "read_evidence":reads,
        "final_ledger_digest":research.ledger.snapshot()["ledger_digest"],
        "real_source_stateful_transitions_inferred":False,
        "scope":"stateless_read_only_resource_observations_not_remote_state_machine",
    }
    return result,research.ledger.snapshot()


def _save(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts")
    parser.add_argument("--live-only",action="store_true")
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if args.live_only:
        report,ledger=real_readonly_experiment()
        _save(out/"trace-derived-live-report.json",report)
        _save(out/"trace-derived-live-ledger.json",ledger)
        print("VL64_TRACE_LIVE "+json.dumps(report,sort_keys=True))
        return
    experiment=controlled_experiment()
    for name,key in (
        ("initial-trace-ledger.json","initial_ledger"),
        ("observed-trace-ledger.json","ledger"),
        ("learned-action-ground.json","ground"),
        ("trace-derived-candidates.json","candidate_catalogue"),
        ("trace-derived-synthesis.json","synthesis"),
        ("trace-driven-session.json","session"),
        ("trace-derived-report.json","report"),
    ):
        _save(out/name,experiment[key])
    intention=ObserverIntention(
        "ground.traces.plan",
        "workspace:"+(out/"initial-trace-ledger.json").as_posix(),
        options={"max_states":2,"max_depth":3,"max_words":64},output="json")
    jump=ObserverIntention(
        "ground.action.navigate",
        "workspace:"+(out/"learned-action-ground.json").as_posix(),
        options={"action":"flip","steps":10**12},output="json")
    _save(out/"trace-synthesis-intention.json",intention.to_dict())
    _save(out/"learned-action-navigation-intention.json",jump.to_dict())
    print("VL64_TRACE_REPORT "+json.dumps(experiment["report"],sort_keys=True))
    print("VL64_TRACE_INITIAL_LEDGER "+json.dumps(
        experiment["initial_ledger"],sort_keys=True,separators=(",",":")))
    print("VL64_TRACE_GROUND "+json.dumps(experiment["ground"],sort_keys=True,
                                         separators=(",",":")))
    print("VL64_TRACE_LEDGER "+json.dumps(experiment["ledger"],sort_keys=True,
                                         separators=(",",":")))


if __name__=="__main__":
    main()
