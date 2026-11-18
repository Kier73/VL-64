"""Observation-preserving finite quotients over validated discrete Grounds.

Reuses VL-64's Myhill-Nerode partition-refinement utility. A quotient preserves
the explicitly requested observable outputs and their deterministic futures;
it deliberately cannot reconstruct bytes erased by a non-singleton partition.
"""
from __future__ import annotations

import json

from vld import domain_digest, myhill_nerode_quotient

from .discrete_ground import DiscreteGroundNavigator, MAX_STEPS


def compile_observable_quotient(discrete, labels):
    """Build a transition-congruent abstraction for a closed finite Ground."""
    if not isinstance(discrete,DiscreteGroundNavigator):
        raise TypeError("Expected admitted discrete Ground driver")
    states=discrete.tokens
    names={state.hex() for state in states}
    if (not isinstance(labels,dict) or set(labels)!=names or
            any(not isinstance(label,str) or not 1<=len(label)<=64
                for label in labels.values())):
        raise ValueError("Supply a bounded observation label for every modeled state")
    graph={}
    total=len(states)
    for i,state in enumerate(states):
        graph[state]=states[i+1] if i+1<total else states[discrete.prefix]
    mapping,erasure=myhill_nerode_quotient(
        states,("advance",),lambda s,a:graph[s],
        lambda s:labels[s.hex()])
    block_count=len(set(mapping.values()))
    if block_count<=0 or block_count>total:
        raise ValueError("Invalid quotient partition")
    classes=[]
    for index in range(block_count):
        members=[state for state in states if mapping[state]==index]
        if not members:
            raise ValueError("Missing quotient block")
        outputs={labels[state.hex()] for state in members}
        targets={mapping[graph[state]] for state in members}
        if len(outputs)!=1 or len(targets)!=1:
            raise ValueError("Proposed quotient fails observation/transition congruence")
        classes.append({
            "label":next(iter(outputs)),
            "next":next(iter(targets)),
            "source_states":len(members),
        })
    descriptor={
        "schema":"vl64.observable-quotient-ground.v1",
        "source_ground_digest":discrete.ground["ground_digest"],
        "scope":"complete-finite-declared-model-and-observer-supplied-labels",
        "initial_block":mapping[states[0]],
        "classes":classes,
        "source_state_count":total,
        "label_contract_digest":domain_digest(
            "vl64.observer-state-label-contract.v1",sorted(labels.items())),
        "reconstruction_claim":{
            "requested_observables_preserved_within_declared_model":True,
            "exact_source_bytes_recoverable_from_quotient_alone":False,
            "unlisted_external_states_proven":False,
        },
    }
    descriptor["ground_digest"]=domain_digest(
        "vl64.observable-quotient-ground.v1",descriptor)
    return {
        "ground":descriptor,
        "measurement":{
            "source_states":total,"quotient_states":block_count,
            "states_merged":total-block_count,
            "erasure_entropy_bits":round(erasure,6),
            "preservation":"observation-and-successor-congruence-on-declared-model",
            "byte_reconstruction_qualified":False,
            "description_bytes":len(json.dumps(
                descriptor,sort_keys=True,separators=(",",":")).encode()),
        },
    }


class ObservableQuotientNavigator:
    """Navigate observer-requested outputs without retaining source tokens."""

    def __init__(self,ground):
        if not isinstance(ground,dict):
            raise ValueError("Expected quotient Ground")
        raw={k:v for k,v in ground.items() if k!="ground_digest"}
        if domain_digest("vl64.observable-quotient-ground.v1",raw)!=ground.get(
                "ground_digest"):
            raise ValueError("Quotient Ground digest mismatch")
        if raw.get("schema")!="vl64.observable-quotient-ground.v1":
            raise ValueError("Unknown quotient schema")
        classes=raw.get("classes")
        initial=raw.get("initial_block")
        if (not isinstance(classes,list) or not 1<=len(classes)<=4096 or
                type(initial) is not int or not 0<=initial<len(classes)):
            raise ValueError("Invalid quotient state count")
        if type(raw.get("source_state_count")) is not int or raw[
                "source_state_count"]<len(classes):
            raise ValueError("Invalid original model size")
        for state in classes:
            if (not isinstance(state,dict) or
                    set(state)!={"label","next","source_states"} or
                    not isinstance(state["label"],str) or
                    not 1<=len(state["label"])<=64 or
                    type(state["next"]) is not int or
                    not 0<=state["next"]<len(classes) or
                    type(state["source_states"]) is not int or
                    state["source_states"]<1):
                raise ValueError("Invalid quotient transition or label")
        if sum(x["source_states"] for x in classes)!=raw["source_state_count"]:
            raise ValueError("Quotient members do not account for source states")
        self.ground=ground
        self.classes=classes
        self.start=initial
        self.trace=[]
        self.cycle_start=None

    def _build_trace(self):
        if self.cycle_start is not None:
            return
        path=[]
        visited={}
        current=self.start
        while current not in visited:
            if len(path)>len(self.classes):
                raise ValueError("Quotient transition graph not closed")
            visited[current]=len(path)
            path.append(current)
            current=self.classes[current]["next"]
        self.trace=path
        self.cycle_start=visited[current]

    def _block_at(self,steps):
        self._build_trace()
        prefix=self.cycle_start
        if steps<len(self.trace):
            return self.trace[steps]
        return self.trace[prefix+(steps-prefix)%(len(self.trace)-prefix)]

    def realize(self,steps,result_type="state.observable",*,
                max_work=10,max_hops=8,include_data=False):
        if type(steps) is not int or not 0<=steps<=MAX_STEPS:
            return {"status":"REFUSED","reason":"INVALID_NAVIGATION_HORIZON"}
        if result_type!="state.observable":
            return {"status":"UNRESOLVED","reason":"QUOTIENT_CANNOT_REALIZE_EXACT_BYTES"}
        if (type(max_work) is not int or max_work<2 or
                type(max_hops) is not int or max_hops<2):
            return {"status":"UNRESOLVED","reason":"NO_ROUTE_WITHIN_CEILINGS"}
        block=self._block_at(steps)
        value=self.classes[block]["label"]
        route={
            "status":"QUALIFIED_MULTI_HOP","estimated_work":2,
            "stages":[
                {"name":"interpret-observable-quotient","from":"ground.description",
                 "to":"initial.block","work":1},
                {"name":"navigate-observable-block","from":"initial.block",
                 "to":"state.observable","work":1},
            ],
        }
        receipt={
            "status":"OBSERVED_GROUND_NAVIGATION",
            "ground_digest":self.ground["ground_digest"],
            "horizon":steps,"route":route,
            "stage_receipts":[
                {"relation":"interpret-observable-quotient",
                 "source_ground_digest":self.ground["source_ground_digest"]},
                {"relation":"navigate-observable-block",
                 "block":block,"output_digest":domain_digest(
                    "vl64.observable-output.v1",value)},
            ],
            "result":{"type":result_type,"value":value,"block":block},
            "interpretation_scope":
                "observer-output-equivalence-on-declared-finite-model-only",
        }
        receipt["receipt_digest"]=domain_digest(
            "vl64.observable-quotient-receipt.v1",receipt)
        return receipt

    def test_new_observations(self,observations):
        samples=list(observations)
        if not 1<=len(samples)<=512:
            raise ValueError("Need fresh indexed observable samples")
        contradictions=[]
        for step,label in samples:
            if (type(step) is not int or not 0<=step<=MAX_STEPS or
                    not isinstance(label,str)):
                raise ValueError("Invalid quotient challenge")
            expected=self.classes[self._block_at(step)]["label"]
            if expected!=label:
                contradictions.append({
                    "index":step,"expected":expected,"observed":label})
        report={
            "schema":"vl64.observable-quotient-challenge.v1",
            "ground_digest":self.ground["ground_digest"],
            "status":"CONTRADICTED" if contradictions
                     else "CONSISTENT_ON_NEW_OBSERVATIONS",
            "contradictions":contradictions[:16],
            "observations":len(samples),
            "scope":"observer-labeled-finite-model-not-exact-source-state",
        }
        report["challenge_digest"]=domain_digest(
            "vl64.observable-quotient-challenge.v1",report)
        return report
