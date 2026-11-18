"""Synthesize small action-conditioned Grounds from actual byte-output traces.

A finite resettable Mealy observation tree is authoritative only for observed
edges. Bounded candidate completion searches small finite state quotients;
every unobserved transition in the candidates is explicitly a conjecture.
No unknown behavior is certified from an incomplete trace.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re

from vld import domain_digest
from .active_ground import (
    MAX_ACTIONS, MAX_OUTPUT, MAX_WORD, ActiveGroundSession,
    QualifiedResetProbe, action_ground, ActionHypothesis,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class TraceLedger:
    """Record action/output observations and preserve conflicting repeats."""

    def __init__(self, actions):
        self.actions=tuple(actions)
        if (not 1<=len(self.actions)<=MAX_ACTIONS or
                len(set(self.actions))!=len(self.actions) or
                any(not isinstance(a,str) or not re.fullmatch(
                    r"[a-z][a-z0-9_.-]{0,47}",a) for a in self.actions)):
            raise ValueError("Invalid action alphabet")
        self.observations=[]
        self.contradictions=[]
        self.edges={}  # prefix -> action -> observed output bytes

    def observe(self, word, outputs, *, source="unattributed",
                policy="external-evidence"):
        if (not isinstance(word,(tuple,list)) or
                not 1<=len(word)<=MAX_WORD or
                any(action not in self.actions for action in word) or
                not isinstance(outputs,(tuple,list)) or
                len(outputs)!=len(word) or
                any(not isinstance(out,bytes) or len(out)>MAX_OUTPUT
                    for out in outputs) or
                not isinstance(source,str) or len(source)>160 or
                not isinstance(policy,str) or len(policy)>80):
            raise ValueError("Invalid bounded action/output observation")
        prefix=()
        changes=[]
        conflicts=[]
        for action,out in zip(word,outputs):
            existing=self.edges.get(prefix,{}).get(action)
            if existing is not None and existing!=out:
                conflicts.append({
                    "prefix":list(prefix),"action":action,
                    "previous_sha256":_sha(existing),
                    "new_sha256":_sha(out)})
            else:
                changes.append((prefix,action,out))
            prefix=prefix+(action,)
        record={
            "schema":"vl64.action-trace-evidence.v1",
            "word":list(word),
            "outputs_hex":[out.hex() for out in outputs],
            "output_sha256":[_sha(out) for out in outputs],
            "source":source,"host_policy":policy,
            "status":"CONTRADICTED_REPEATED_OBSERVATION" if conflicts else
                     "OBSERVED_ACTION_TRACE",
            "conflicts":conflicts,
            "scope":"observed_resettable_action_word_not_unobserved_edges",
        }
        record["evidence_digest"]=domain_digest(
            "vl64.action-trace-evidence.v1",record)
        self.observations.append(record)
        if conflicts:
            self.contradictions.append(record)
            return record
        for prefix,action,out in changes:
            self.edges.setdefault(prefix,{})[action]=out
        return record

    def query(self,host,word):
        if (not isinstance(host,QualifiedResetProbe) or
                host.actions!=self.actions):
            raise ValueError("Host is not qualified for this action alphabet")
        observed=host.run(word)
        return self.observe(word,observed,source=host.source,policy=host.policy)

    @classmethod
    def from_snapshot(cls,record):
        """Replay a serialized evidence ledger, checking every observation.

        A digest protects integrity of the description but cannot authenticate
        a host's claimed source or prove that a physical action really ran.
        """
        if (not isinstance(record,dict) or set(record)!={
                "schema","actions","observations","contradiction_count",
                "scope","ledger_digest"} or
                record["schema"]!="vl64.action-trace-ledger.v1" or
                not isinstance(record["observations"],list) or
                len(record["observations"])>256):
            raise ValueError("Unsupported or oversized trace ledger")
        if domain_digest("vl64.action-trace-ledger.v1",{
                k:v for k,v in record.items() if k!="ledger_digest"})!=record[
                    "ledger_digest"]:
            raise ValueError("Trace ledger content digest mismatch")
        ledger=cls(record["actions"])
        for saved in record["observations"]:
            if (not isinstance(saved,dict) or
                    not isinstance(saved.get("outputs_hex"),list)):
                raise ValueError("Malformed trace evidence object")
            try:
                values=tuple(bytes.fromhex(v) for v in saved["outputs_hex"])
                              if all(isinstance(v,str) for v in saved["outputs_hex"])
                              else None
            except ValueError as exc:
                raise ValueError("Malformed byte output representation") from exc
            if values is None or any(v.hex()!=encoded for v,encoded in zip(
                    values,saved["outputs_hex"])):
                raise ValueError("Noncanonical byte output representation")
            generated=ledger.observe(saved.get("word"),values,
                                     source=saved.get("source"),
                                     policy=saved.get("host_policy"))
            if generated!=saved:
                raise ValueError("Observed trace receipt is not independently reconstructible")
        if ledger.snapshot()!=record:
            raise ValueError("Trace ledger history does not reconstruct exactly")
        return ledger

    def traces(self):
        return [
            (tuple(record["word"]),
             tuple(bytes.fromhex(h) for h in record["outputs_hex"]))
            for record in self.observations
            if record["status"]=="OBSERVED_ACTION_TRACE"
        ]

    def snapshot(self):
        result={
            "schema":"vl64.action-trace-ledger.v1",
            "actions":list(self.actions),
            "observations":self.observations,
            "contradiction_count":len(self.contradictions),
            "scope":"resettable_observed_paths_only",
        }
        result["ledger_digest"]=domain_digest("vl64.action-trace-ledger.v1",result)
        return result


def _tree(ledger):
    nodes={()}
    transitions={}
    symbols=set()
    for word,outputs in ledger.traces():
        prefix=()
        for action,out in zip(word,outputs):
            child=prefix+(action,)
            nodes.add(child)
            existing=transitions.get((prefix,action))
            if existing is not None and existing!=(child,out):
                raise ValueError("Inconsistent observed action trie")
            transitions[(prefix,action)]=(child,out)
            symbols.add(out)
            prefix=child
    ordered=sorted(nodes,key=lambda p:(len(p),p))
    return ordered,transitions,sorted(symbols)


class TraceGroundSynthesizer:
    """Bounded deterministic state-merging and unknown-edge completions.

    Candidates are NOT asserted to be the only consistent machines when
    enumeration is capped, output alphabet is incomplete or state bound is
    chosen by the caller. Unknown transitions are explicitly hypothetical.
    """

    def __init__(self,*,max_states=2,max_candidates=24,
                 max_assignments=15000,max_nodes=24):
        if (type(max_states) is not int or not 1<=max_states<=4 or
                type(max_candidates) is not int or not 2<=max_candidates<=32 or
                type(max_assignments) is not int or not 50<=max_assignments<=100000 or
                type(max_nodes) is not int or not 2<=max_nodes<=40):
            raise ValueError("Unsafe synthesizer bounds")
        self.max_states=max_states
        self.max_candidates=max_candidates
        self.max_assignments=max_assignments
        self.max_nodes=max_nodes

    def discover(self,ledger):
        if not isinstance(ledger,TraceLedger):
            raise TypeError("Expected exact TraceLedger")
        if ledger.contradictions:
            return {"status":"CONTRADICTED_RESET_DETERMINISM",
                    "reason":"REPEATED_ACTION_PREFIX_HAS_DIFFERENT_OUTPUTS",
                    "ledger_digest":ledger.snapshot()["ledger_digest"],
                    "contradiction_digests":[r["evidence_digest"]
                                             for r in ledger.contradictions],
                    "candidates":[]}
        nodes,edges,alphabet=_tree(ledger)
        if not edges:
            return {"status":"UNRESOLVED","reason":"NO_ACTION_OBSERVATIONS",
                    "candidates":[]}
        if len(nodes)>self.max_nodes or len(alphabet)>8:
            return {"status":"UNRESOLVED","reason":"OBSERVATION_TREE_BUDGET",
                    "observed_nodes":len(nodes),"observed_output_symbols":len(alphabet),
                    "candidates":[]}
        actions=ledger.actions
        # A descriptor may contain guessed output bytes, but only outputs from
        # actual observations in this ledger are allowed as guess symbols.
        candidates=[]
        records=[]
        known_digests=set()
        examined=0
        truncated=False
        paths={prefix:index for index,prefix in enumerate(nodes)}
        max_per_size=max(2,(self.max_candidates//self.max_states)+1)
        for n_states in range(1,self.max_states+1):
            before=len(candidates)
            limit=min(self.max_candidates,before+max_per_size)
            def add_completed(labels,rules):
                nonlocal examined,truncated
                missing=[(q,a) for q in range(n_states) for a in actions
                         if (q,a) not in rules]
                table=dict(rules)
                def fill(i):
                    nonlocal examined,truncated
                    if len(candidates)>=limit or examined>=self.max_assignments:
                        truncated=True
                        return
                    if i==len(missing):
                        states=[]
                        for q in range(n_states):
                            row={}
                            for action in actions:
                                successor,out=table[q,action]
                                row[action]={"next":successor,
                                             "output_hex":out.hex()}
                            states.append(row)
                        desc=action_ground(actions,states)
                        digest=desc["ground_digest"]
                        if digest not in known_digests:
                            known_digests.add(digest)
                            candidates.append(desc)
                            records.append({
                                "ground_digest":digest,
                                "states":n_states,
                                "observed_state_action_edges":len(rules),
                                "hypothesized_state_action_edges":len(missing),
                                "source":"constrained_trace_completion",
                                "unseen_transitions_certified":False,
                            })
                        return
                    q,action=missing[i]
                    for next_state in range(n_states):
                        for out in alphabet:
                            examined+=1
                            if examined>self.max_assignments:
                                truncated=True
                                return
                            table[q,action]=(next_state,out)
                            fill(i+1)
                            if len(candidates)>=limit or examined>=self.max_assignments:
                                break
                        if len(candidates)>=limit or examined>=self.max_assignments:
                            break
                    table.pop((q,action),None)
                fill(0)

            def assign(i,labels,rule,used):
                nonlocal examined,truncated
                if len(candidates)>=limit or examined>=self.max_assignments:
                    truncated=True
                    return
                if i==len(nodes):
                    if used==n_states:
                        add_completed(labels,rule)
                    return
                node=nodes[i]
                parent=node[:-1]
                action=node[-1]
                edge=edges[(parent,action)]
                out=edge[1]
                from_state=labels[paths[parent]]
                for state in range(min(used,n_states-1)+1):
                    examined+=1
                    if examined>self.max_assignments:
                        truncated=True
                        return
                    previous=rule.get((from_state,action))
                    if previous is not None and previous!=(state,out):
                        continue
                    amended=dict(rule)
                    amended[(from_state,action)]=(state,out)
                    assign(i+1,labels+[state],amended,
                           max(used,state+1))
                    if len(candidates)>=limit or examined>=self.max_assignments:
                        return

            # BFS/canonical observed-node state assignments give a candidate
            # state numbering; the root always has state 0.
            assign(1,[0],{},1)
            if truncated and examined>=self.max_assignments:
                break
            if len(candidates)>=self.max_candidates:
                break
        checked=[]
        for descriptor,meta in zip(candidates,records):
            model=ActionHypothesis(descriptor)
            if not all(model.predict(word)==outputs
                       for word,outputs in ledger.traces()):
                raise RuntimeError("Synthesized candidate contradicts known bytes")
            checked.append(meta)
        report={
            "status":"SYNTHESIZED_BOUNDED_CANDIDATES" if candidates else "UNRESOLVED",
            "schema":"vl64.trace-to-ground-synthesis.v1",
            "ledger_digest":ledger.snapshot()["ledger_digest"],
            "actions":list(actions),
            "observed_prefix_tree_nodes":len(nodes),
            "observed_action_edges":len(edges),
            "observed_output_symbols_hex":[v.hex() for v in alphabet],
            "max_states":self.max_states,
            "assignments_evaluated":examined,
            "search_truncated":truncated,
            "candidate_count":len(candidates),
            "candidate_metadata":checked,
            "candidates":candidates,
            "scope":"enumerated_consistent_small_models_with_unknown_edge_guesses",
            "completeness_claim":False,
        }
        report["synthesis_digest"]=domain_digest(
            "vl64.trace-to-ground-synthesis.v1",report)
        return report


class TraceDrivenDiscovery:
    """Observe, synthesize, discriminate, and incorporate each counterexample."""

    def __init__(self,host,*,max_states=2,max_candidates=24):
        if not isinstance(host,QualifiedResetProbe):
            raise TypeError("Host must be a qualified resettable oracle")
        self.host=host
        self.ledger=TraceLedger(host.actions)
        self.synthesizer=TraceGroundSynthesizer(
            max_states=max_states,max_candidates=max_candidates)
        self.rounds=[]

    def seed(self,words):
        for word in words:
            self.ledger.query(self.host,word)
        return self.synthesizer.discover(self.ledger)

    def step(self,*,max_depth=3,max_words=128,max_cost=12):
        proposal=self.synthesizer.discover(self.ledger)
        if proposal["status"]!="SYNTHESIZED_BOUNDED_CANDIDATES":
            return {"status":"UNRESOLVED","synthesis":proposal}
        if len(proposal["candidates"])<2:
            return {"status":"UNRESOLVED","reason":"INSUFFICIENT_DISTINCT_CANDIDATES",
                    "synthesis":proposal}
        session=ActiveGroundSession(proposal["candidates"])
        selected=session.plan(max_depth=max_depth,max_words=max_words,
                              max_cost=max_cost)
        if selected["status"]!="QUALIFIED_INFORMATIVE_PROBE":
            return {"status":"UNRESOLVED","reason":"NO_PROBE_DISTINGUISHES_GENERATED_CANDIDATES",
                    "synthesis":proposal,"plan":selected}
        try:
            actual=self.host.run(selected["word"])
        except (ValueError,RuntimeError,OSError) as exc:
            return {"status":"UNRESOLVED","reason":"HOST_PROBE_FAILED",
                    "detail":type(exc).__name__}
        narrowed=session.apply(
            selected,actual,probe_source=self.host.source,
            authority=self.host.policy)
        recorded=self.ledger.observe(
            selected["word"],actual,source=self.host.source,
            policy=self.host.policy)
        if recorded["status"]!="OBSERVED_ACTION_TRACE":
            return {"status":"CONTRADICTED_RESET_DETERMINISM",
                    "evidence":recorded}
        updated=self.synthesizer.discover(self.ledger)
        receipt={
            "schema":"vl64.trace-driven-active-round.v1",
            "status":"RESYNTHESIZED_FROM_NEW_EVIDENCE" if updated[
                "status"]=="SYNTHESIZED_BOUNDED_CANDIDATES" else
                "UNRESOLVED",
            "previous_synthesis_digest":proposal["synthesis_digest"],
            "selected_plan":selected,
            "observed_action_evidence":recorded,
            "old_candidates_remaining":len(session.models),
            "old_hypotheses_contradicted":narrowed[
                "status"]=="CONTRADICTED_ALL_HYPOTHESES",
            "new_candidate_count":updated.get("candidate_count",0),
            "updated_synthesis_digest":updated.get("synthesis_digest"),
            "scope":"bounded_resynthesis_from_real_observed_bytes",
        }
        receipt["round_digest"]=domain_digest(
            "vl64.trace-driven-active-round.v1",receipt)
        self.rounds.append(receipt)
        return receipt

    def receipt(self):
        report={
            "schema":"vl64.trace-driven-research-session.v1",
            "ledger":self.ledger.snapshot(),
            "rounds":self.rounds,
            "latest_synthesis":self.synthesizer.discover(self.ledger),
            "scope":"finite_model_version_space_no_global_equivalence_claim",
        }
        report["session_digest"]=domain_digest(
            "vl64.trace-driven-research-session.v1",report)
        return report
