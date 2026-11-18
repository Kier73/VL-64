"""Bounded active discrimination of registered action-conditioned byte Grounds.

Each candidate is a complete deterministic finite *hypothesis* over an explicit
action alphabet. The planner picks a resettable action word by expected version-
space information gain. A host-installed read-only probe supplies actual bytes.
Disagreement preserves evidence; nothing installs executable code from a Ground.

This is version-space experiment selection, not full L* or proof of equivalence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass

from vld import domain_digest

MAX_MODELS=32
MAX_STATES=128
MAX_ACTIONS=8
MAX_OUTPUT=256
MAX_WORD=6
MAX_CANDIDATES=2048


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(record):
    return len(json.dumps(record,sort_keys=True,
                          separators=(",",":"),allow_nan=False).encode())


def action_ground(actions, states, *, initial_state=0):
    """Package a host-authored candidate with a checkable structural digest."""
    record={
        "schema":"vl64.action-conditioned-ground.v1",
        "scope":"host-declared-complete-finite-deterministic-candidate",
        "actions":list(actions),
        "initial_state":initial_state,
        "states":states,
    }
    record["ground_digest"]=domain_digest("vl64.action-conditioned-ground.v1",record)
    # This validation rejects any remote/host-supplied executable payload.
    ActionHypothesis(record)
    return record


class ActionHypothesis:
    def __init__(self,ground):
        if not isinstance(ground,dict):
            raise ValueError("Candidate must be a typed Ground")
        record={k:v for k,v in ground.items() if k!="ground_digest"}
        if domain_digest("vl64.action-conditioned-ground.v1",record)!=ground.get(
                "ground_digest"):
            raise ValueError("Action Ground digest mismatch")
        if set(record)!={"schema","scope","actions","initial_state","states"} or (
                record["schema"]!="vl64.action-conditioned-ground.v1"):
            raise ValueError("Unknown action model contract")
        actions=record["actions"]
        states=record["states"]
        initial=record["initial_state"]
        if (not isinstance(actions,list) or not 1<=len(actions)<=MAX_ACTIONS or
                len(set(actions))!=len(actions) or any(
                    not isinstance(a,str) or not re.fullmatch(
                        r"[a-z][a-z0-9_.-]{0,47}",a) for a in actions)):
            raise ValueError("Invalid action alphabet")
        if (not isinstance(states,list) or not 1<=len(states)<=MAX_STATES or
                type(initial) is not int or not 0<=initial<len(states)):
            raise ValueError("Invalid finite state count or initial state")
        self.actions=tuple(actions)
        self.transitions=[]
        for state in states:
            if not isinstance(state,dict) or set(state)!=set(actions):
                raise ValueError("Incomplete action/state transition relation")
            row={}
            for action in actions:
                cell=state[action]
                if (not isinstance(cell,dict) or
                        set(cell)!={"next","output_hex"} or
                        type(cell["next"]) is not int or
                        not 0<=cell["next"]<len(states) or
                        not isinstance(cell["output_hex"],str) or
                        len(cell["output_hex"])>2*MAX_OUTPUT):
                    raise ValueError("Malformed action transition")
                try:
                    output=bytes.fromhex(cell["output_hex"])
                except ValueError as exc:
                    raise ValueError("Output is not hexadecimal bytes") from exc
                if output.hex()!=cell["output_hex"]:
                    raise ValueError("Noncanonical output hex")
                row[action]=(cell["next"],output)
            self.transitions.append(row)
        self.initial=initial
        self.ground=ground
        self.digest=ground["ground_digest"]

    def predict(self,word):
        state=self.initial
        outputs=[]
        for action in word:
            if action not in self.actions:
                raise ValueError("Unregistered action in model")
            state,out=self.transitions[state][action]
            outputs.append(out)
        return tuple(outputs)


class ActionGroundNavigator:
    """Pure state-model navigation, not actual external actuation.

    For a repeated single action, powers of the finite successor map give the
    terminal modeled state in O(|states| log n). Only the terminal-step output
    is materialized; a trillion intermediate outputs are not claimed.
    """

    def __init__(self,ground):
        self.model=ActionHypothesis(ground)
        self.ground=ground

    def repeat(self,action,steps):
        if action not in self.model.actions:
            return {"status":"UNRESOLVED","reason":"UNREGISTERED_ACTION"}
        if type(steps) is not int or not 0<=steps<=10**18:
            return {"status":"REFUSED","reason":"INVALID_ACTION_HORIZON"}
        model=self.model
        size=len(model.transitions)
        transition=tuple(model.transitions[state][action][0]
                         for state in range(size))
        # A finite-state transformation is a compact computational relation;
        # power composition avoids materializing predecessor histories.
        def compose(left,right):
            return tuple(left[right[i]] for i in range(size))
        result=tuple(range(size))
        power=transition
        remaining=steps
        composed=0
        while remaining:
            if remaining&1:
                result=compose(power,result)
                composed+=1
            power=compose(power,power)
            remaining>>=1
            composed+=1
        state=result[model.initial]
        last=None
        if steps:
            # Find predecessor at t-1 with one more bounded power calculation
            # rather than keeping the whole path.
            prefix=tuple(range(size))
            power=transition
            remaining=steps-1
            while remaining:
                if remaining&1:
                    prefix=compose(power,prefix)
                power=compose(power,power)
                remaining>>=1
            before=prefix[model.initial]
            last=model.transitions[before][action][1]
        receipt={
            "schema":"vl64.action-model-navigation.v1",
            "status":"MODELED_ACTION_NAVIGATION",
            "ground_digest":model.digest,
            "action":action,
            "steps":steps,
            "terminal_state":state,
            "last_output_hex":last.hex() if last is not None else None,
            "last_output_sha256":_digest(last) if last is not None else None,
            "model_transformation_size":size,
            "power_compositions":composed,
            "scope":"hypothetical-deterministic-model-not-remote-action-execution",
        }
        receipt["receipt_digest"]=domain_digest(
            "vl64.action-model-navigation.v1",receipt)
        return receipt



class QualifiedResetProbe:
    """A host-installed callback; no observer input may install one."""

    def __init__(self, actions, reset_and_run, *,
                 policy="local-simulated", source="host-authorized"):
        if (not callable(reset_and_run) or
                policy not in ("local-simulated","read-only-remote") or
                not isinstance(source,str) or not source or len(source)>160):
            raise ValueError("Unsupported observer-host probe authority")
        self.actions=tuple(actions)
        if (not 1<=len(self.actions)<=MAX_ACTIONS or
                len(set(self.actions))!=len(self.actions) or
                any(not isinstance(a,str) or not re.fullmatch(
                    r"[a-z][a-z0-9_.-]{0,47}",a) for a in self.actions)):
            raise ValueError("Invalid host action alphabet")
        self.reset_and_run=reset_and_run
        self.policy=policy
        self.source=source

    def run(self,word):
        if (not isinstance(word,(list,tuple)) or not 1<=len(word)<=MAX_WORD or
                any(a not in self.actions for a in word)):
            raise ValueError("Unqualified probe action word")
        raw=self.reset_and_run(tuple(word))
        if (not isinstance(raw,(tuple,list)) or len(raw)!=len(word) or
                any(not isinstance(x,bytes) or len(x)>MAX_OUTPUT for x in raw)):
            raise ValueError("Host did not supply complete bounded byte outputs")
        return tuple(raw)


def _partitions(models,word):
    groups={}
    for model in models:
        signature=tuple(x.hex() for x in model.predict(word))
        groups.setdefault(signature,[]).append(model.digest)
    count=len(models)
    expected_entropy=sum(
        (len(group)/count)*math.log2(len(group))
        for group in groups.values() if len(group)>1
    )
    return groups,math.log2(count)-expected_entropy


class ActiveGroundSession:
    def __init__(self, candidates, *, costs=None):
        if (not isinstance(candidates,(list,tuple)) or
                not 2<=len(candidates)<=MAX_MODELS):
            raise ValueError("Need 2..32 candidate Ground descriptions")
        self.models=[ActionHypothesis(c) for c in candidates]
        alphabet=self.models[0].actions
        if (any(model.actions!=alphabet for model in self.models) or
                len({model.digest for model in self.models})!=len(self.models)):
            raise ValueError("Candidate actions must match and digests must be unique")
        self.actions=alphabet
        self.costs=dict(costs or {action:1 for action in alphabet})
        if (set(self.costs)!=set(alphabet) or
                any(type(value) is not int or not 1<=value<=100
                    for value in self.costs.values())):
            raise ValueError("Invalid caller-declared action costs")
        self.initial_digests=tuple(model.digest for model in self.models)
        self.evidence=[]
        self.status="HYPOTHESES_UNRESOLVED"

    def plan(self,*,max_depth=4,max_words=512,max_cost=12):
        if (type(max_depth) is not int or not 1<=max_depth<=MAX_WORD or
                type(max_words) is not int or not 1<=max_words<=MAX_CANDIDATES or
                type(max_cost) is not int or not 1<=max_cost<=600):
            raise ValueError("Invalid active experiment search limits")
        examined=0
        best=None
        for depth in range(1,max_depth+1):
            for word in itertools.product(self.actions,repeat=depth):
                if examined>=max_words:
                    break
                cost=sum(self.costs[action] for action in word)
                if cost>max_cost:
                    continue
                examined+=1
                groups,gain=_partitions(self.models,word)
                if gain<=1e-12:
                    continue
                score=gain/cost
                rank=(-score,-gain,cost,len(word),word)
                if best is None or rank<best[0]:
                    best=(rank,word,cost,groups,gain)
            if examined>=max_words:
                break
        if best is None:
            return {
                "status":"UNRESOLVED",
                "reason":"NO_DISCRIMINATOR_WITHIN_SEARCH_BUDGET",
                "examined_words":examined,
                "candidate_count":len(self.models),
                "exhaustive_within_declared_depth":examined<max_words,
            }
        _,word,cost,groups,gain=best
        plan={
            "schema":"vl64.active-ground-probe-plan.v1",
            "status":"QUALIFIED_INFORMATIVE_PROBE",
            "word":list(word),
            "predicted_partitions":[{
                "outputs_hex":list(signature),
                "hypothesis_digests":sorted(members),
            } for signature,members in sorted(groups.items())],
            "expected_information_bits":round(gain,6),
            "score_bits_per_declared_cost":round(gain/cost,6),
            "declared_cost":cost,
            "candidate_count":len(self.models),
            "examined_words":examined,
            "exhaustive_within_declared_depth":examined<max_words,
            "evidence_scope":"uniform-candidate-prior-and-host-declared-costs",
        }
        plan["plan_digest"]=domain_digest("vl64.active-ground-probe-plan.v1",plan)
        return plan

    def apply(self,plan,observed,*,probe_source="unattributed",
              authority="externally-recorded"):
        if (not isinstance(plan,dict) or
                plan.get("status")!="QUALIFIED_INFORMATIVE_PROBE"):
            raise ValueError("No qualified planned experiment")
        digest=plan.get("plan_digest")
        if domain_digest("vl64.active-ground-probe-plan.v1",{
                k:v for k,v in plan.items() if k!="plan_digest"})!=digest:
            raise ValueError("Probe plan digest mismatched")
        word=plan["word"]
        if (not isinstance(observed,(tuple,list)) or len(observed)!=len(word) or
                any(not isinstance(x,bytes) or len(x)>MAX_OUTPUT for x in observed)):
            raise ValueError("Observation must contain every bounded action output")
        current={model.digest for model in self.models}
        plan_members={d for part in plan["predicted_partitions"]
                      for d in part["hypothesis_digests"]}
        if (plan_members!=current or plan["candidate_count"]!=len(current)):
            raise ValueError("Stale experiment plan cannot update candidate Ground")
        survivors=[m for m in self.models if m.predict(word)==tuple(observed)]
        record={
            "schema":"vl64.active-ground-observation.v1",
            "plan_digest":digest,
            "action_word":list(word),
            "observed_outputs_hex":[x.hex() for x in observed],
            "observed_output_digests":[_digest(x) for x in observed],
            "observer_source":probe_source,
            "observer_authority":authority,
            "before_hypothesis_digests":sorted(current),
            "after_hypothesis_digests":sorted(m.digest for m in survivors),
            "eliminated_hypothesis_digests":sorted(
                current-{m.digest for m in survivors}),
            "status":("CONTRADICTED_ALL_HYPOTHESES" if not survivors else
                      "ONE_HYPOTHESIS_REMAINS" if len(survivors)==1 else
                      "HYPOTHESES_REFINED"),
            "scope":"qualified-recorded-output-not-physical-equivalence-proof",
        }
        record["observation_digest"]=domain_digest(
            "vl64.active-ground-observation.v1",record)
        self.models=survivors
        self.evidence.append(record)
        self.status=record["status"]
        return record

    def step(self,host,*,max_depth=4,max_words=512,max_cost=12):
        if not isinstance(host,QualifiedResetProbe) or host.actions!=self.actions:
            raise ValueError("Host has not admitted this action alphabet")
        if not self.models:
            return {"status":"CONTRADICTED_ALL_HYPOTHESES"}
        if len(self.models)==1:
            return {"status":"ONE_HYPOTHESIS_REMAINS"}
        plan=self.plan(max_depth=max_depth,max_words=max_words,max_cost=max_cost)
        if plan["status"]!="QUALIFIED_INFORMATIVE_PROBE":
            return plan
        try:
            outputs=host.run(plan["word"])
        except (ValueError,RuntimeError,OSError) as exc:
            return {"status":"UNRESOLVED","reason":"PROBE_FAILED",
                    "detail":type(exc).__name__,"plan":plan}
        observation=self.apply(plan,outputs,probe_source=host.source,
                               authority=host.policy)
        return {"status":observation["status"],"plan":plan,
                "observation":observation}

    def receipt(self):
        result={
            "schema":"vl64.active-ground-session.v1",
            "status":self.status,
            "initial_ground_digests":list(self.initial_digests),
            "remaining_ground_digests":[m.digest for m in self.models],
            "observations":self.evidence,
            "scope":"finite-candidate-elimination-not-general-model-learning",
        }
        result["session_digest"]=domain_digest(
            "vl64.active-ground-session.v1",result)
        return result
