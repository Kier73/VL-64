"""Use VL-64 to actively choose informative actions, not blindly iterate them.

Controlled experiment: four hidden finite deterministic machines share current
responses; two resettable probes distinguish them. The independent oracle is
implemented separately from the candidate tables. A modeled action Ground can
then navigate 10^12 repeated steps without executing that many actions.

Optional live experiment: read-only GitHub HTTPS observations, explicitly a
stateless resource-identity test and NOT inferred remote stateful dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vld import ObserverIntention, domain_digest
from .active_ground import (
    ActionGroundNavigator, ActiveGroundSession,QualifiedResetProbe,action_ground,
)
from .sources import acquire


ACTIONS=("advance","inspect","switch")
PINNED_REVISION="e0cb4d5dfd21c018a00e9f0c95f5d73263472955"


def _cell(nxt,output):
    return {"next":nxt,"output_hex":output.hex()}


def candidate_models():
    models=[]
    for i in range(4):
        x=b"x" if i in (0,1) else b"y"
        y=b"left" if i in (0,2) else b"right"
        transitions=[
            {"advance":_cell(1,b"tick"),"inspect":_cell(0,b"ready"),
             "switch":_cell(2,b"tick")},
            {"advance":_cell(1,b"tick"),"inspect":_cell(1,x),
             "switch":_cell(1,b"tick")},
            {"advance":_cell(2,b"tick"),"inspect":_cell(2,y),
             "switch":_cell(2,b"tick")},
        ]
        models.append(action_ground(ACTIONS,transitions))
    return models


def independent_host(hidden_variant=2):
    # Deliberately NOT calling model.predict: this represents an independently
    # instrumented resettable program with the same externally visible actions.
    def run(word):
        state=0
        result=[]
        for action in word:
            if state==0:
                if action=="inspect":
                    value=b"ready"
                else:
                    state=1 if action=="advance" else 2
                    value=b"tick"
            elif state==1:
                value=(b"x" if hidden_variant in (0,1) else b"y"
                       ) if action=="inspect" else b"tick"
            else:
                value=(b"left" if hidden_variant in (0,2) else b"right"
                       ) if action=="inspect" else b"tick"
            result.append(value)
        return tuple(result)
    return QualifiedResetProbe(ACTIONS,run,source="controlled-resettable-oracle")


def controlled_experiment():
    models=candidate_models()
    session=ActiveGroundSession(models)
    hidden=independent_host(2)
    start=time.perf_counter_ns()
    rounds=[]
    while len(session.models)>1 and len(rounds)<4:
        recorded=session.step(hidden,max_depth=2,max_words=32)
        rounds.append(recorded)
        if recorded["status"] not in ("HYPOTHESES_REFINED",
                                      "ONE_HYPOTHESIS_REMAINS"):
            raise RuntimeError("Active oracle did not supply a qualified observation")
    learn_ms=(time.perf_counter_ns()-start)/1e6
    if len(session.models)!=1 or session.models[0].digest!=models[2]["ground_digest"]:
        raise RuntimeError("Active experiment did not identify the hidden candidate")
    ground=session.models[0].ground
    navigate=ActionGroundNavigator(ground)
    horizon=10**12
    modeled=navigate.repeat("advance",horizon)
    if modeled["status"]!="MODELED_ACTION_NAVIGATION" or modeled[
            "terminal_state"]!=1 or modeled["power_compositions"]>128:
        raise RuntimeError("Powered action navigation failed")
    # Independent bounded stepping reference.
    ref=independent_host(2).run(("advance",)*6)
    if ref[-1].hex()!=navigate.repeat("advance",6)["last_output_hex"]:
        raise RuntimeError("Modeled action output conflicts with independent oracle")
    return {
        "candidates":{
            "schema":"vl64.action-candidate-catalogue.v1",
            "candidates":models,
            "costs":{name:1 for name in ACTIONS},
        },
        "selected_ground":ground,
        "session":session.receipt(),
        "report":{
            "schema":"vl64.active-action-ground-lab.v1",
            "status":"ONE_HYPOTHESIS_REMAINS",
            "candidate_count_initial":4,
            "candidate_count_final":len(session.models),
            "resettable_probe_rounds":len(rounds),
            "action_words":[r["plan"]["word"] for r in rounds],
            "information_gain_per_round_bits":[
                r["plan"]["expected_information_bits"] for r in rounds],
            "total_observed_action_outputs":sum(
                len(r["observation"]["observed_outputs_hex"]) for r in rounds),
            "selected_ground_digest":ground["ground_digest"],
            "modeled_repeated_action_horizon":horizon,
            "modeled_terminal_state":modeled["terminal_state"],
            "power_compositions":modeled["power_compositions"],
            "training_ms_one_run":round(learn_ms,4),
            "scope":"controlled-independent-resettable-synthetic-machine",
            "future_external_state_proven":False,
            "core_source_modified":False,
        }
    }


def read_only_internet_experiment():
    """Real public GitHub bytes as read-only action outputs, no remote writes."""
    root=Path.cwd()
    origin="https://raw.githubusercontent.com/Kier73/VL-64/"+PINNED_REVISION
    source_map={
        "read_goals":origin+"/docs/RESEARCH_GOALS.md",
        "read_readme":origin+"/README.md",
    }
    def read(action):
        capture=acquire(source_map[action],root=root,max_bytes=65536)
        if not capture["complete"]:
            raise ValueError("Remote read-only probe observed an incomplete resource")
        return hashlib.sha256(capture["data"]).digest(),{
            "url":source_map[action],
            "observed_sha256":hashlib.sha256(capture["data"]).hexdigest(),
            "observed_bytes":len(capture["data"]),
            "capture_status":"COMPLETE",
            "source_scope":"public-pinned-revision-read-only",
        }
    initial={}
    read_evidence=[]
    for action in sorted(source_map):
        result,evidence=read(action)
        initial[action]=result
        read_evidence.append(evidence)
    if initial["read_goals"]==initial["read_readme"]:
        raise ValueError("Source digests unexpectedly identical")
    def make_model(readme_digest):
        return action_ground(sorted(source_map),[{
            "read_goals":_cell(0,initial["read_goals"]),
            "read_readme":_cell(0,readme_digest),
        }])
    true_model=make_model(initial["read_readme"])
    competitor=make_model(initial["read_goals"])
    observed=[]
    def read_host(word):
        out=[]
        for action in word:
            value,evidence=read(action)
            observed.append(evidence)
            out.append(value)
        return tuple(out)
    qualified=QualifiedResetProbe(
        sorted(source_map),read_host,policy="read-only-remote",
        source="pinned-public-github-raw-read-only")
    session=ActiveGroundSession([true_model,competitor])
    result=session.step(qualified,max_depth=1,max_words=2)
    return {
        "schema":"vl64.active-web-byte-observation.v1",
        "status":result["status"],
        "scope":"read-only-resource-identity-discrimination-not-remote-stateful-automata-learning",
        "source_revision":PINNED_REVISION,
        "selected_actions":result.get("plan",{}).get("word"),
        "initial_byte_observations":read_evidence,
        "selected_byte_observations":observed,
        "model_consistency_digests":[m.digest for m in session.models],
        "host_policy":qualified.policy,
        "evidence":session.receipt(),
    }


def _write_json(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts")
    parser.add_argument("--live-only",action="store_true")
    args=parser.parse_args()
    path=Path(args.out)
    path.mkdir(exist_ok=True,parents=True)
    if args.live_only:
        result=read_only_internet_experiment()
        _write_json(path/"active-live-web-observation.json",result)
        print("VL64_ACTIVE_WEB "+json.dumps(result,sort_keys=True))
        if result["status"]!="ONE_HYPOTHESIS_REMAINS":
            raise SystemExit("Real read-only candidate could not be identified")
        return
    result=controlled_experiment()
    _write_json(path/"active-candidates.json",result["candidates"])
    _write_json(path/"active-selected-ground.json",result["selected_ground"])
    _write_json(path/"active-session.json",result["session"])
    _write_json(path/"active-results.json",result["report"])
    plan=ObserverIntention(
        "ground.active.plan",
        "workspace:"+(path/"active-candidates.json").as_posix(),
        options={"max_depth":2,"max_words":32},output="json")
    navigation=ObserverIntention(
        "ground.action.navigate",
        "workspace:"+(path/"active-selected-ground.json").as_posix(),
        options={"action":"advance","steps":10**12},output="json")
    _write_json(path/"active-next-probe-intent.json",plan.to_dict())
    _write_json(path/"active-action-navigation-intent.json",navigation.to_dict())
    print("VL64_ACTIVE_REPORT "+json.dumps(result["report"],sort_keys=True))
    print("VL64_ACTIVE_CANDIDATES "+json.dumps(result["candidates"],
                                             sort_keys=True,separators=(",",":")))
    print("VL64_ACTIVE_SELECTED "+json.dumps(result["selected_ground"],
                                           sort_keys=True,separators=(",",":")))
    print("VL64_ACTIVE_SESSION "+json.dumps(result["session"],
                                          sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
