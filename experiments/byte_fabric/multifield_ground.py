"""Evidence-qualified, multi-field byte Grounds with partial observations.

Learn the fixed byte template, several disjoint varying spans, and a sparse
deterministic action transition graph from EXACT before/action/after frames.
Unseen state/action pairs remain UNKNOWN; never complete them by assumption.
A partial byte observation can locate compatible states and recommend the
missing field with highest distinguishing information per observed byte.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import defaultdict

from vld import domain_digest

MAX_FRAME=65536
MAX_SPANS=16
MAX_STATES=256
MAX_ACTIONS=16
MAX_EDGES=4096
MAX_HORIZON=10**18


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _packed(raw):
    return base64.b64encode(raw).decode("ascii")


def _bytes(text):
    return base64.b64decode(text,validate=True)


def _size(value):
    return len(json.dumps(value,sort_keys=True,separators=(",",":"),
                          allow_nan=False).encode())


def _transition_record(item):
    if (not isinstance(item,(tuple,list)) or len(item)!=3 or
            not isinstance(item[0],bytes) or
            not isinstance(item[1],str) or
            not isinstance(item[2],bytes)):
        raise ValueError("Expected exact (before:bytes, action:str, after:bytes)")
    before,action,after=item
    if (not 1<=len(before)<=MAX_FRAME or len(before)!=len(after) or
            not action or len(action)>48 or
            not all(c.isalnum() or c in "_.-" for c in action) or
            not action[0].isalpha()):
        raise ValueError("Unqualified byte frame or action")
    return before,action,after


def _spans(positions):
    spans=[]
    for pos in positions:
        if not spans or pos>spans[-1]["offset"]+spans[-1]["width"]:
            spans.append({"offset":pos,"width":1})
        else:
            spans[-1]["width"]+=1
    return spans


def _token(frame,spans):
    return b"".join(frame[x["offset"]:x["offset"]+x["width"]]
                    for x in spans)


def _template(frame,spans):
    mutable=bytearray(frame)
    for field in spans:
        start=field["offset"]
        mutable[start:start+field["width"]]=b"\0"*field["width"]
    return bytes(mutable)


def _render(template,spans,token):
    result=bytearray(template)
    position=0
    for field in spans:
        start,width=field["offset"],field["width"]
        result[start:start+width]=token[position:position+width]
        position+=width
    return bytes(result)


def _dependency_witnesses(training,spans):
    """Correlated observations only: witnessed sensitivity, not causality."""
    edges={}
    grouped=defaultdict(list)
    for before,action,after in training:
        grouped[action].append((before,after))
    for action,observations in grouped.items():
        for i,(first,out_first) in enumerate(observations):
            for other,out_other in observations[i+1:]:
                changed=[
                    k for k,span in enumerate(spans)
                    if first[span["offset"]:span["offset"]+span["width"]] !=
                    other[span["offset"]:span["offset"]+span["width"]]
                ]
                if len(changed)!=1:
                    continue
                source=changed[0]
                for target,span in enumerate(spans):
                    if out_first[span["offset"]:span["offset"]+span["width"]] != (
                            out_other[span["offset"]:span["offset"]+span["width"]]):
                        key=(action,source,target)
                        edges.setdefault(key, {
                            "action":action,
                            "input_span":source,
                            "output_span":target,
                            "kind":"witnessed_conditional_sensitivity_not_causal_proof",
                            "witness_before_sha256":[_sha(first),_sha(other)],
                            "witness_after_sha256":[_sha(out_first),_sha(out_other)],
                        })
    return [edges[k] for k in sorted(edges)]


def discover_multifield_ground(training,heldout=()):
    """Build a partial action graph; refuse any conflicting known transition."""
    training=[_transition_record(x) for x in training]
    heldout=[_transition_record(x) for x in heldout]
    if not 2<=len(training)<=MAX_EDGES or len(heldout)>MAX_EDGES:
        raise ValueError("Need bounded observed transition evidence")
    frames=[frame for before,_,after in training for frame in (before,after)]
    length=len(frames[0])
    if any(len(frame)!=length for frame in frames):
        raise ValueError("Inconsistent training frame widths")
    baseline=frames[0]
    changed=[
        i for i in range(length)
        if any(frame[i]!=baseline[i] for frame in frames[1:])
    ]
    if not changed:
        raise ValueError("No state-variable bytes observed")
    spans=_spans(changed)
    if not 2<=len(spans)<=MAX_SPANS:
        raise ValueError("Need 2..16 witnessed distinct variable byte spans")
    if any(field["width"]>32 for field in spans):
        raise ValueError("One inferred field exceeds 32 bytes")
    template=_template(baseline,spans)
    for frame in frames:
        if _template(frame,spans)!=template:
            raise ValueError("Inconsistent constant byte context")
    tokens=sorted({_token(frame,spans) for frame in frames})
    if len(tokens)>MAX_STATES:
        raise ValueError("Too many explicit observed states")
    ids={token:i for i,token in enumerate(tokens)}
    actions=sorted({action for _,action,_ in training})
    if len(actions)>MAX_ACTIONS:
        raise ValueError("Too many observed actions")
    known={}
    for before,action,after in training:
        key=(ids[_token(before,spans)],action)
        end=ids[_token(after,spans)]
        if key in known and known[key]!=end:
            raise ValueError("CONTRADICTED_OBSERVED_ACTION_TRANSITION")
        known[key]=end
    edges=[
        {"from_state":source,"action":action,"to_state":target}
        for (source,action),target in sorted(known.items())
    ]
    agreements=[]
    unresolved=[]
    for before,action,after in heldout:
        if len(before)!=length or len(after)!=length:
            raise ValueError("Heldout frame size mismatch")
        if (_template(before,spans)!=template or
                _template(after,spans)!=template):
            raise ValueError("Heldout frame contradicts static Ground bytes")
        source=ids.get(_token(before,spans))
        destination=ids.get(_token(after,spans))
        if source is None or action not in actions or (source,action) not in known:
            unresolved.append({"action":action,"before_sha256":_sha(before),
                               "after_sha256":_sha(after),
                               "reason":"UNOBSERVED_TRANSITION"})
        elif destination!=known[(source,action)]:
            raise ValueError("HELDOUT_TRANSITION_CONTRADICTS_GROUND")
        else:
            agreements.append({"action":action,"before_sha256":_sha(before),
                               "after_sha256":_sha(after)})
    ground={
        "schema":"vl64.multifield-action-ground.v1",
        "scope":"observed-fixed-layout-sparse-action-graph-with-unknown-edges",
        "frame_length":length,
        "template_b64":_packed(template),
        "variable_spans":spans,
        "state_tokens_hex":[token.hex() for token in tokens],
        "actions":actions,
        "observed_edges":edges,
        "dependency_witnesses":_dependency_witnesses(training,spans),
        "training_evidence_digest":domain_digest(
            "vl64.multifield-training-evidence.v1",
            [(_sha(before),action,_sha(after))
             for before,action,after in training]),
        "heldout_evidence_digest":domain_digest(
            "vl64.multifield-heldout-evidence.v1",
            [(_sha(before),action,_sha(after))
             for before,action,after in heldout]),
        "heldout_verified_count":len(agreements),
        "heldout_unresolved":unresolved,
        "unseen_transitions_proven":False,
    }
    ground["ground_digest"]=domain_digest(
        "vl64.multifield-action-ground.v1",ground)
    description_bytes=_size(ground)
    return {
        "ground":ground,
        "measurement":{
            "raw_transition_frame_bytes":sum(
                len(before)+len(after) for before,_,after in training),
            "description_bytes":description_bytes,
            "description_ratio":round(
                sum(len(before)+len(after) for before,_,after in training)/
                description_bytes,4),
            "fields":len(spans),"states":len(tokens),"actions":len(actions),
            "observed_edges":len(edges),
            "unobserved_action_edges":len(tokens)*len(actions)-len(edges),
            "verified_heldout_count":len(agreements),
            "unresolved_heldout_count":len(unresolved),
            "scope":"fixed_layout_structural_memory_with_partial_transition_knowledge",
        },
    }


class MultiFieldNavigator:
    """Exact known-edge navigation and three-valued partial observation."""

    def __init__(self,ground):
        if not isinstance(ground,dict):
            raise ValueError("Expected multi-field Ground")
        raw={k:v for k,v in ground.items() if k!="ground_digest"}
        if domain_digest("vl64.multifield-action-ground.v1",raw)!=ground.get(
                "ground_digest"):
            raise ValueError("Multifield Ground digest mismatch")
        if raw.get("schema")!="vl64.multifield-action-ground.v1":
            raise ValueError("Unsupported multifield Ground schema")
        n=raw.get("frame_length")
        fields=raw.get("variable_spans")
        states=raw.get("state_tokens_hex")
        actions=raw.get("actions")
        edges=raw.get("observed_edges")
        if (type(n) is not int or not 1<=n<=MAX_FRAME or
                not isinstance(fields,list) or not 2<=len(fields)<=MAX_SPANS or
                not isinstance(states,list) or not 1<=len(states)<=MAX_STATES or
                not isinstance(actions,list) or not 1<=len(actions)<=MAX_ACTIONS or
                not isinstance(edges,list) or len(edges)>MAX_EDGES):
            raise ValueError("Invalid multi-field Ground shape")
        last=0
        for field in fields:
            if (not isinstance(field,dict) or
                    set(field)!={"offset","width"} or
                    type(field["offset"]) is not int or
                    type(field["width"]) is not int or
                    not last<=field["offset"]<n or
                    not 1<=field["width"]<=32 or
                    field["offset"]+field["width"]>n):
                raise ValueError("Invalid variable span layout")
            last=field["offset"]+field["width"]
        try:
            template=_bytes(raw["template_b64"])
            tokens=tuple(bytes.fromhex(token) for token in states)
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError("Malformed Ground encoded bytes") from exc
        width=sum(x["width"] for x in fields)
        if (len(template)!=n or _template(template,fields)!=template or
                any(len(token)!=width for token in tokens) or
                len(set(tokens))!=len(tokens) or
                [x.hex() for x in tokens]!=states):
            raise ValueError("Malformed Ground states or static byte context")
        if (len(set(actions))!=len(actions) or
                not all(isinstance(x,str) and 0<len(x)<=48 for x in actions)):
            raise ValueError("Malformed action alphabet")
        known={}
        for edge in edges:
            if (not isinstance(edge,dict) or
                    set(edge)!={"from_state","to_state","action"} or
                    type(edge["from_state"]) is not int or
                    type(edge["to_state"]) is not int or
                    not 0<=edge["from_state"]<len(tokens) or
                    not 0<=edge["to_state"]<len(tokens) or
                    edge["action"] not in actions):
                raise ValueError("Malformed finite transition")
            key=(edge["from_state"],edge["action"])
            if key in known:
                raise ValueError("Duplicate action edge")
            known[key]=edge["to_state"]
        self.ground=ground
        self.template=template
        self.spans=tuple(fields)
        self.states=tokens
        self.actions=tuple(actions)
        self.transitions=known
        self.ids={token:i for i,token in enumerate(tokens)}

    def materialize(self,state_id):
        if type(state_id) is not int or not 0<=state_id<len(self.states):
            raise ValueError("Unknown model state")
        return _render(self.template,self.spans,self.states[state_id])

    def _intervals(self,known_intervals):
        if (not isinstance(known_intervals,(list,tuple)) or
                not 1<=len(known_intervals)<=MAX_FRAME):
            raise ValueError("Need one or more known byte intervals")
        valid=[]
        last=0
        for interval in known_intervals:
            if (not isinstance(interval,(tuple,list)) or len(interval)!=2 or
                    any(type(x) is not int for x in interval) or
                    not last<=interval[0]<interval[1]<=len(self.template)):
                raise ValueError("Intervals must be ordered, non-overlapping and inside frame")
            valid.append(tuple(interval))
            last=interval[1]
        return valid

    def observe_partial(self,observed,known_intervals):
        if not isinstance(observed,bytes) or len(observed)!=len(self.template):
            raise ValueError("Partial observation must preserve indexed frame width")
        intervals=self._intervals(known_intervals)
        known=tuple(i for a,b in intervals for i in range(a,b))
        matches=[]
        for state in range(len(self.states)):
            candidate=self.materialize(state)
            if all(candidate[i]==observed[i] for i in known):
                matches.append(state)
        base={
            "schema":"vl64.multifield-partial-observation.v1",
            "ground_digest":self.ground["ground_digest"],
            "known_intervals":[list(x) for x in intervals],
            "known_byte_count":len(known),
            "known_content_sha256":_sha(bytes(observed[i] for i in known)),
            "compatible_model_states":matches,
            "model_state_count":len(self.states),
            "scope":"observed_positions_only_unobserved_bytes_unverified",
        }
        if not matches:
            base.update({"status":"CONTRADICTED","next_probe":None})
        elif len(matches)==1:
            base.update({"status":"UNIQUE_COMPATIBLE_MODEL_STATE",
                         "next_probe":None,
                         "state_id":matches[0]})
        else:
            choices=[]
            for index,field in enumerate(self.spans):
                start,stop=field["offset"],field["offset"]+field["width"]
                if all(i in known for i in range(start,stop)):
                    continue
                values=[self.states[state][sum(x["width"] for x in self.spans[:index]):
                              sum(x["width"] for x in self.spans[:index+1])]
                        for state in matches]
                groups=defaultdict(int)
                for item in values:
                    groups[item]+=1
                prior=math.log2(len(matches))
                expected=sum((v/len(matches))*math.log2(v) for v in
                             groups.values())
                gain=prior-expected
                if gain>1e-12:
                    choices.append((-gain/field["width"],-gain,field["width"],
                                    index,start,stop))
            if choices:
                _,negative_gain,width,index,start,stop=min(choices)
                base.update({"status":"MULTIPLE_COMPATIBLE_MODEL_STATES",
                             "next_probe":{
                                 "field_index":index,
                                 "byte_interval":[start,stop],
                                 "expected_information_bits":round(-negative_gain,5),
                                 "cost_scope":"bytes_observed_with_uniform_state_prior",
                             }})
            else:
                base.update({"status":"INDETERMINATE_COMPATIBLE_STATES",
                             "next_probe":None})
        base["observation_digest"]=domain_digest(
            "vl64.multifield-partial-observation.v1",base)
        return base

    def navigate(self,initial_state_hex,action,steps,
                 result_type="bytes.snapshot",include_data=False):
        if (not isinstance(initial_state_hex,str) or
                not isinstance(action,str) or
                type(steps) is not int or not 0<=steps<=MAX_HORIZON or
                type(include_data) is not bool):
            return {"status":"REFUSED","reason":"INVALID_NAVIGATION_PARAMETERS"}
        if action not in self.actions:
            return {"status":"UNRESOLVED","reason":"UNOBSERVED_ACTION"}
        if result_type not in ("bytes.snapshot","digest.sha256","state.token"):
            return {"status":"UNRESOLVED","reason":"UNSUPPORTED_RESULT_TYPE"}
        try:
            token=bytes.fromhex(initial_state_hex)
        except ValueError:
            return {"status":"REFUSED","reason":"INVALID_STATE_HEX"}
        state=self.ids.get(token)
        if state is None:
            return {"status":"UNRESOLVED","reason":"UNKNOWN_INITIAL_STATE"}
        visited={}
        clock=0
        skipped=0
        while clock<steps:
            if state in visited:
                cycle=clock-visited[state]
                if cycle:
                    repeats=(steps-clock)//cycle
                    clock+=repeats*cycle
                    skipped+=repeats*cycle
                    if clock==steps:
                        break
            else:
                visited[state]=clock
            next_id=self.transitions.get((state,action))
            if next_id is None:
                return {
                    "status":"UNRESOLVED","reason":"UNOBSERVED_STATE_ACTION",
                    "completed_steps":clock,
                    "missing":{"state_id":state,"action":action},
                    "skipped_modeled_steps":skipped,
                    "ground_digest":self.ground["ground_digest"],
                }
            state=next_id
            clock+=1
        frame=self.materialize(state)
        result=({"type":result_type,"hex":self.states[state].hex()} if
                result_type=="state.token" else
                {"type":result_type,"value":_sha(frame)} if
                result_type=="digest.sha256" else
                {"type":result_type,"sha256":_sha(frame),"length":len(frame)})
        if result_type=="bytes.snapshot" and include_data and len(frame)<=4096:
            result["data_b64"]=_packed(frame)
        receipt={
            "schema":"vl64.multifield-navigation.v1",
            "status":"OBSERVED_GROUND_NAVIGATION",
            "ground_digest":self.ground["ground_digest"],
            "action":action,"horizon":steps,"terminal_state_id":state,
            "cycle_skipped_steps":skipped,
            "evaluated_transitions":clock-skipped,
            "result":result,
            "interpretation_scope":"navigated_recorded_action_graph_not_proven_future_source_behavior",
        }
        receipt["receipt_digest"]=domain_digest(
            "vl64.multifield-navigation.v1",receipt)
        return receipt

    def challenge(self,observations):
        if not isinstance(observations,(list,tuple)) or not 1<=len(observations)<=512:
            raise ValueError("Need bounded new independent transition observations")
        checked=[]
        mismatches=[]
        unknown=[]
        for item in observations:
            before,action,after=_transition_record(item)
            if (len(before)!=len(self.template) or
                    len(after)!=len(self.template)):
                raise ValueError("Challenge frame width mismatch")
            source=self.ids.get(_token(before,self.spans))
            dest=self.ids.get(_token(after,self.spans))
            relation=self.transitions.get((source,action))
            record={"before_sha256":_sha(before),"action":action,
                    "after_sha256":_sha(after)}
            if (_template(before,self.spans)!=self.template or
                    _template(after,self.spans)!=self.template):
                mismatches.append({**record,"reason":"STATIC_TEMPLATE_CHANGED"})
            elif relation is None:
                unknown.append({**record,"reason":"UNKNOWN_TRANSITION"})
            elif relation!=dest:
                mismatches.append({**record,"reason":"CONTRADICTED_TRANSITION"})
            else:
                checked.append(record)
        answer={
            "schema":"vl64.multifield-independent-challenge.v1",
            "status":"CONTRADICTED" if mismatches else
                     "INDETERMINATE" if unknown else
                     "CONSISTENT_ON_CHALLENGED_OBSERVATIONS",
            "ground_digest":self.ground["ground_digest"],
            "consistent":len(checked),
            "mismatches":mismatches[:16],
            "unknown":unknown[:16],
            "observation_evidence_digest":domain_digest(
                "vl64.multifield-challenge-evidence.v1",
                [(_sha(x[0]),x[1],_sha(x[2])) for x in observations]),
            "scope":"checked_known_edges_only_no_universal_future_proof",
        }
        answer["challenge_digest"]=domain_digest(
            "vl64.multifield-independent-challenge.v1",answer)
        return answer
