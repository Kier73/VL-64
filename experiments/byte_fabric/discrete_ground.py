"""A finite, non-affine Virtual Ground learned from byte-state observations.

The compact runtime's ObservedGround checks the observed transition relation.
This external adapter adds a *closed-cycle* compression certificate and an
O(1) projection for a witnessed preperiod + deterministic finite cycle.

A closed cycle in finite samples is a model hypothesis, not a theorem that an
unknown external process will continue to obey it. Counterexamples are retained.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Sequence

from vld import ObservedGround, domain_digest

MAX_SNAPSHOT = 65536
MAX_STATES = 4096
MAX_STEPS = 10**18


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _size(data: dict) -> int:
    return len(json.dumps(data, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode())


def _frames(value):
    sequence = list(value)
    if not sequence or len(sequence) > 10000 or not all(
            isinstance(frame, bytes) and
            1 <= len(frame) <= MAX_SNAPSHOT for frame in sequence):
        raise ValueError("Expected a bounded sequence of byte snapshots")
    if any(len(frame) != len(sequence[0]) for frame in sequence):
        raise ValueError("Snapshots have incompatible widths")
    return sequence


def _observed_span(training, heldout, max_width):
    initial = training[0]
    changed = [position for position in range(len(initial))
               if any(frame[position] != initial[position] for frame in training[1:])]
    if not changed:
        raise ValueError("No witnessed changing state")
    offset, stop = changed[0], changed[-1] + 1
    if (stop - offset > max_width or stop-offset != len(changed)):
        raise ValueError("Changing bytes do not form one admitted contiguous span")
    template = initial[:offset] + b"\x00" * (stop-offset) + initial[stop:]
    for frame in training + heldout:
        if frame[:offset] + b"\x00" * (stop-offset) + frame[stop:] != template:
            raise ValueError("Observed byte template changed outside learned span")
    return offset, stop-offset, template


def discover_discrete_ground(training, heldout, *, max_states=MAX_STATES,
                             max_field_width=32):
    """Infer and validate a finite preperiod/cycle Ground from observed bytes.

    Requires two *witnessed* complete cycle traversals in training, plus
    separate held-out observations. Refuses ambiguous successor relations.
    """
    if (type(max_states) is not int or not 2 <= max_states <= MAX_STATES or
            type(max_field_width) is not int or not 1 <= max_field_width <= 64):
        raise ValueError("Invalid finite Ground discovery ceiling")
    training, heldout = _frames(training), _frames(heldout)
    if len(training) < 8 or len(heldout) < 3:
        raise ValueError("Need at least 8 training and 3 held-out frames")
    if len(training)+len(heldout) > 10000:
        raise ValueError("Too many observed frames")
    if len(training[0]) != len(heldout[0]):
        raise ValueError("Held-out snapshots have incompatible widths")
    offset, width, template = _observed_span(
        training, heldout, max_field_width)
    def token(frame):
        return frame[offset:offset+width]
    tokens = [token(frame) for frame in training]
    seen = {}
    repeat = None
    for index, state in enumerate(tokens):
        if state in seen:
            repeat = (seen[state], index - seen[state])
            break
        seen[state] = index
    if repeat is None:
        raise ValueError("No witnessed closed finite-state transition")
    start, length = repeat
    if (length <= 0 or start + length > max_states or
            len(tokens) < start + 2*length + 1):
        raise ValueError("Need two observed cycles and bounded states")
    unique = tokens[:start+length]
    if len(set(unique)) != len(unique):
        raise ValueError("Finite model has ambiguous state identities")
    def at(index):
        return unique[index if index < start
                      else start + (index-start) % length]
    if any(at(index) != state for index,state in enumerate(tokens)):
        raise ValueError("Observed cycle transitions conflict")
    # The compact core independently audits that each projected source has
    # only one observed successor and that long-horizon cycle navigation works.
    core = ObservedGround({"token":tuple(range(offset,offset+width))}).fit(
        zip(training[:-1], training[1:]))
    for original in unique:
        probe = bytearray(template)
        probe[offset:offset+width] = original
        result = core.predict(bytes(probe), "token")
        if result["status"] != "SINGLE_OBSERVED_SUCCESSOR":
            raise ValueError("ObservedGround reports uncertain successor")
    if core.navigate(training[0],len(training)-1,"token")[
            "terminal_state"] != tokens[-1].hex():
        raise ValueError("Compact-core transition navigation disagrees")
    for step,frame in enumerate(heldout, start=len(training)):
        expected = template[:offset] + at(step) + template[offset+width:]
        if frame != expected:
            raise ValueError("Held-out transition mismatch at "+str(step))
    descriptor={
        "schema":"vl64.discrete-orbit-ground.v1",
        "scope":"finite-closed-observed-orbit-hypothesis",
        "byte_template_b64":base64.b64encode(template).decode("ascii"),
        "template_bytes":len(template),
        "variable_offset":offset,
        "variable_width":width,
        "preperiod_states":start,
        "cycle_states":length,
        "state_tokens_b64":base64.b64encode(b"".join(unique)).decode("ascii"),
        "training_frames":len(training),
        "heldout_frames":len(heldout),
        "training_evidence_digest":domain_digest(
            "vl64.discrete-ground-training.v1", [_hash(f) for f in training]),
        "heldout_evidence_digest":domain_digest(
            "vl64.discrete-ground-validation.v1", [_hash(f) for f in heldout]),
        "claims":{
            "closed_successor_relation_witnessed":True,
            "complete_arbitrary_machine_model":False,
            "unobserved_future_behavior_proven":False,
            "action_alphabet":"one_unlabeled_time_step",
        },
    }
    descriptor["ground_digest"]=domain_digest(
        "vl64.discrete-orbit-ground.v1", descriptor)
    stored=_size(descriptor)
    raw=sum(len(f) for f in training+heldout)
    return {
        "ground":descriptor,
        "measurement":{
            "raw_observed_bytes":raw,
            "ground_description_bytes":stored,
            "compression_ratio":round(raw/stored,3),
            "witnessed_cycle_states":length,
            "witnessed_prefix_states":start,
            "heldout_exact_frames":len(heldout),
            "future_behavior_proven":False,
        },
    }


class DiscreteGroundNavigator:
    """Navigate learned finite orbits without iterating their transitions."""

    def __init__(self, ground):
        if not isinstance(ground,dict):
            raise ValueError("Expected a discrete Ground descriptor")
        raw={k:v for k,v in ground.items() if k!="ground_digest"}
        if domain_digest("vl64.discrete-orbit-ground.v1",raw) != ground.get(
                "ground_digest"):
            raise ValueError("Discrete Ground digest mismatch")
        if raw.get("schema")!="vl64.discrete-orbit-ground.v1":
            raise ValueError("Unsupported discrete Ground schema")
        offset,width=raw.get("variable_offset"),raw.get("variable_width")
        prefix,cycle=raw.get("preperiod_states"),raw.get("cycle_states")
        size=raw.get("template_bytes")
        if any(type(x) is not int for x in (offset,width,prefix,cycle,size)):
            raise ValueError("Ground shape metadata must be integers")
        if (not 0 <= offset < size <= MAX_SNAPSHOT or
                not 1 <= width <= 64 or offset+width > size or
                not 0 <= prefix or not 1 <= cycle or
                prefix+cycle > MAX_STATES):
            raise ValueError("Ground shape metadata outside bounds")
        try:
            template=base64.b64decode(raw["byte_template_b64"],validate=True)
            joined=base64.b64decode(raw["state_tokens_b64"],validate=True)
        except (ValueError,KeyError) as exc:
            raise ValueError("Malformed encoded discrete Ground") from exc
        if (len(template)!=size or
                template[offset:offset+width] != b"\x00"*width or
                len(joined)!=(prefix+cycle)*width):
            raise ValueError("Inconsistent compressed Ground byte lengths")
        tokens=tuple(joined[i:i+width] for i in range(0,len(joined),width))
        if len(set(tokens))!=len(tokens):
            raise ValueError("Duplicate token breaks unique finite states")
        self.ground=ground
        self.template=template
        self.tokens=tokens
        self.offset,self.width,self.prefix,self.cycle=offset,width,prefix,cycle

    def _token_at(self,step):
        return self.tokens[step if step < self.prefix
                           else self.prefix+(step-self.prefix)%self.cycle]

    def _snapshot(self,step):
        return (self.template[:self.offset] + self._token_at(step)
                +self.template[self.offset+self.width:])

    def route(self,result_type,max_work=10,max_hops=8):
        targets={"state.token":2,"bytes.snapshot":3,"digest.sha256":4}
        if result_type not in targets:
            return {"status":"UNRESOLVED","reason":"UNSUPPORTED_RESULT_TYPE"}
        if (type(max_work) is not int or not 1 <= max_work <= 10000 or
                type(max_hops) is not int or not 1 <= max_hops <= 8):
            return {"status":"REFUSED","reason":"INVALID_ROUTE_CEILING"}
        required=targets[result_type]
        if max_work < required or max_hops < required:
            return {"status":"UNRESOLVED","reason":"NO_ROUTE_WITHIN_CEILINGS"}
        types=["observed.bytes","initial.token","terminal.token",
               "realized.bytes","result.digest"]
        stages=["interpret-token","navigate-closed-cycle",
                "materialize-token","digest-realization"]
        return {
            "status":"QUALIFIED_MULTI_HOP","estimated_work":required,
            "stages":[{"name":name,"from":types[i],"to":types[i+1],"work":1}
                      for i,name in enumerate(stages[:required])],
        }

    def realize(self,steps,result_type="bytes.snapshot",*,max_work=10,
                max_hops=8,include_data=False):
        if type(steps) is not int or not 0 <= steps <= MAX_STEPS:
            return {"status":"REFUSED","reason":"INVALID_NAVIGATION_HORIZON"}
        if type(include_data) is not bool:
            return {"status":"REFUSED","reason":"INVALID_OUTPUT_OPTION"}
        route=self.route(result_type,max_work,max_hops)
        if route["status"]!="QUALIFIED_MULTI_HOP":
            return route
        original=self.tokens[0]
        token=self._token_at(steps)
        snapshot=self._snapshot(steps)
        values=[self.template[:self.offset]+original+self.template[
            self.offset+self.width:],original,token,snapshot,_hash(snapshot)]
        receipts=[]
        for index,stage in enumerate(route["stages"]):
            receipts.append({
                "relation":stage["name"],"from":stage["from"],"to":stage["to"],
                "input_digest":domain_digest("vl64.discrete-stage-input.v1",
                                             values[index]),
                "output_digest":domain_digest("vl64.discrete-stage-output.v1",
                                              values[index+1]),
            })
        if result_type=="state.token":
            result={"type":result_type,"hex":token.hex(),"byte_count":len(token)}
        elif result_type=="bytes.snapshot":
            result={"type":result_type,"sha256":_hash(snapshot),
                    "length":len(snapshot)}
            if include_data and len(snapshot)<=4096:
                result["data_b64"]=base64.b64encode(snapshot).decode("ascii")
        else:
            result={"type":result_type,"value":_hash(snapshot)}
        report={
            "status":"OBSERVED_GROUND_NAVIGATION",
            "ground_digest":self.ground["ground_digest"],
            "horizon":steps,"route":route,"stage_receipts":receipts,
            "result":result,
            "model_index":(steps if steps<self.prefix
                           else self.prefix+(steps-self.prefix)%self.cycle),
            "navigation_complexity":"O(1)-cycle-index-after-model-load",
            "interpretation_scope":
                "witnessed-closed-cycle-hypothesis-not-proven-remote-future",
        }
        report["receipt_digest"]=domain_digest(
            "vl64.discrete-navigation-receipt.v1", report)
        return report

    def test_new_observations(self,observations):
        frames=list(observations)
        if not 1<=len(frames)<=512:
            raise ValueError("Need 1..512 fresh indexed observations")
        contradictions=[]
        for step,raw in frames:
            if (type(step) is not int or not 0<=step<=MAX_STEPS or
                    not isinstance(raw,bytes) or len(raw)!=len(self.template)):
                raise ValueError("Invalid independent challenge frame")
            predicted=self._snapshot(step)
            if predicted!=raw:
                contradictions.append({
                    "index":step,"observed_sha256":_hash(raw),
                    "predicted_sha256":_hash(predicted),
                })
        evidence={
            "schema":"vl64.discrete-ground-challenge.v1",
            "ground_digest":self.ground["ground_digest"],
            "status":"CONTRADICTED" if contradictions else
                     "CONSISTENT_ON_NEW_OBSERVATIONS",
            "observed_frames":len(frames),
            "contradictions":contradictions[:16],
            "observation_digest":domain_digest(
                "vl64.discrete-challenge-observations.v1",
                [(i,_hash(raw)) for i,raw in frames]),
            "scope":"indexed-observations-only-not-universal-machine-equivalence",
        }
        evidence["challenge_digest"]=domain_digest(
            "vl64.discrete-challenge.v1",evidence)
        return evidence

    def check_against_core(self,training,steps):
        """Cross-check compact ObservedGround deterministic cycle navigator."""
        frames=_frames(training)
        core=ObservedGround({"token":tuple(range(
            self.offset,self.offset+self.width))}).fit(zip(frames[:-1],frames[1:]))
        # Not all training sets necessarily witness a complete closed cycle;
        # the fitted Ground requires two complete traversals, so this one does.
        return core.navigate(frames[0],steps,"token")[
            "terminal_state"]==self._token_at(steps).hex()
