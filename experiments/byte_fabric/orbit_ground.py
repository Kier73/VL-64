"""Compressive executable Grounds and typed multi-hop state navigation.

A witnessed sequence of same-shaped byte snapshots can be represented by one
byte template, a typed scalar lens, and an empirically validated affine law.
That description can be navigated without replaying the intervening snapshots.
Only the witnessed/held-out observations are verified, not all future states.
"""
from __future__ import annotations

import base64
import hashlib
import heapq
import json
import math
from dataclasses import dataclass

from vld import affine_navigate, domain_digest

MAX_SNAPSHOT = 65536
MAX_HOPS = 8
MAX_STEPS = 10**18


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_size(obj):
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))


@dataclass(frozen=True)
class SpanInterpreter:
    """Fixed byte-span scalar interpretation; other types can be host adapters."""

    offset: int
    width: int
    encoding: str = "u-le"

    def __post_init__(self):
        if (type(self.offset) is not int or self.offset < 0 or
                type(self.width) is not int or not 1 <= self.width <= 16 or
                self.encoding not in ("u-le", "u-be", "ascii-decimal") or
                (self.encoding != "ascii-decimal" and self.width > 8)):
            raise ValueError("Invalid span interpretation")

    @property
    def capacity(self):
        return 10**self.width if self.encoding == "ascii-decimal" else 1 << (8*self.width)

    def decode(self, frame):
        if not isinstance(frame, bytes) or len(frame) < self.offset + self.width:
            raise ValueError("Snapshot missing interpreted span")
        part = frame[self.offset:self.offset+self.width]
        if self.encoding == "ascii-decimal":
            if any(n < 48 or n > 57 for n in part):
                raise ValueError("Snapshot contains nondecimal field")
            return int(part.decode("ascii"))
        return int.from_bytes(part, "little" if self.encoding == "u-le" else "big")

    def encode(self, value):
        if type(value) is not int or not 0 <= value < self.capacity:
            raise ValueError("State exceeds codec representation")
        if self.encoding == "ascii-decimal":
            return f"{value:0{self.width}d}".encode("ascii")
        return value.to_bytes(self.width, "little" if self.encoding == "u-le" else "big")

    def blank(self, frame):
        if len(frame) < self.offset + self.width:
            raise ValueError("Snapshot shorter than span")
        return (frame[:self.offset] + b"\0" * self.width
                + frame[self.offset+self.width:])

    def to_dict(self):
        return {"offset":self.offset, "width":self.width, "encoding":self.encoding}


def _fit_affine(states, modulus):
    """Solve modular linear equations; refuse multiple valid coefficients."""
    if type(modulus) is not int or not 2 <= modulus <= (1 << 64):
        raise ValueError("Modulus outside admitted range")
    if len(states) < 5 or len(set(states[:-1])) < 3:
        raise ValueError("Insufficient independent training observations")
    if any(type(x) is not int or x < 0 or x >= modulus for x in states):
        raise ValueError("Observed state outside declared modulus")
    x0, y0 = states[0], states[1]
    # Prefer a pair whose x-difference is invertible mod m; otherwise solve
    # the bounded residue class. Never silently choose one of many models.
    hypotheses = None
    for x, y in zip(states[1:-1], states[2:]):
        delta = (x - x0) % modulus
        rhs = (y - y0) % modulus
        g = math.gcd(delta, modulus)
        if rhs % g:
            raise ValueError("No affine rule satisfies the observed pairs")
        candidate_count = g
        if candidate_count > 4096:
            continue
        reduced = modulus // g
        a0 = (rhs // g * pow(delta // g, -1, reduced)) % reduced if reduced > 1 else 0
        candidates = [
            (a0 + k * reduced) % modulus for k in range(candidate_count)]
        if hypotheses is None or len(candidates) < len(hypotheses):
            hypotheses = candidates
        if len(hypotheses) == 1:
            break
    if hypotheses is None:
        raise ValueError("Underdetermined modular coefficient; decline large search")
    valid = []
    for a in hypotheses:
        b = (y0 - a * x0) % modulus
        if all((a*x+b)%modulus == y for x,y in zip(states[:-1], states[1:])):
            valid.append((a,b))
    if len(valid) != 1:
        raise ValueError("Observed transitions do not identify a unique affine Ground")
    return valid[0]


def infer_ground(training, heldout, interpreter, modulus):
    """Fit on training; validate *all* held-out snapshots and exact template."""
    if not isinstance(interpreter, SpanInterpreter):
        raise TypeError("A host-qualified SpanInterpreter is required")
    training, heldout = list(training), list(heldout)
    if len(training) < 5 or len(heldout) < 3:
        raise ValueError("At least five train and three held-out frames required")
    frames = training + heldout
    if (not isinstance(frames[0], bytes) or
            not 1 <= len(frames[0]) <= MAX_SNAPSHOT):
        raise ValueError("Invalid snapshot")
    baseline = interpreter.blank(frames[0])
    values = []
    for frame in frames:
        if (not isinstance(frame, bytes) or len(frame) != len(frames[0]) or
                interpreter.blank(frame) != baseline):
            raise ValueError("Snapshot differs outside interpreter's variable span")
        values.append(interpreter.decode(frame))
    if modulus > interpreter.capacity:
        raise ValueError("Modulus exceeds interpreter capacity")
    a,b = _fit_affine(values[:len(training)], modulus)
    previous = values[len(training)-1]
    for index,state in enumerate(values[len(training):]):
        predicted = (a*previous+b)%modulus
        if predicted != state:
            raise ValueError("Held-out state transition disagrees at index "+str(index))
        previous = state
    descriptor = {
        "schema": "vl64.compressive-orbit-ground.v1",
        "scope": "single-fixed-template-affine-observed-hypothesis",
        "interpreter": interpreter.to_dict(),
        "template_b64": base64.b64encode(baseline).decode("ascii"),
        "template_length": len(baseline),
        "initial_state": values[0],
        "modulus": modulus,
        "a": a,"b":b,
        "training_frames": len(training),"heldout_frames":len(heldout),
        "training_evidence_digest": domain_digest(
            "vl64.orbit-training-evidence.v1", [_digest(x) for x in training]),
        "heldout_evidence_digest": domain_digest(
            "vl64.orbit-heldout-evidence.v1", [_digest(x) for x in heldout]),
    }
    descriptor["ground_digest"] = domain_digest(
        "vl64.compressive-orbit-ground.v1", descriptor)
    budget = _json_size(descriptor)
    original = sum(len(x) for x in frames)
    return {
        "ground": descriptor,
        "measurement": {
            "source_snapshot_bytes":original,
            "descriptor_bytes":budget,
            "ratio_source_to_descriptor":round(original/budget, 4),
            "representation": "exact-fixed-template-plus-witnessed-affine-transition",
            "compression_scope": "observed-sequence-reconstructible-if-law-holds",
            "heldout_frames_checked": len(heldout),
            "future_states_proven": False,
        },
    }


def _load_ground(ground):
    if not isinstance(ground, dict):
        raise ValueError("Expected Ground object")
    raw={key:value for key,value in ground.items() if key != "ground_digest"}
    if domain_digest("vl64.compressive-orbit-ground.v1",raw) != ground.get("ground_digest"):
        raise ValueError("Ground descriptor digest mismatch")
    if raw.get("schema") != "vl64.compressive-orbit-ground.v1":
        raise ValueError("Unknown compressed Ground schema")
    codec=SpanInterpreter(**raw["interpreter"])
    template=base64.b64decode(raw["template_b64"],validate=True)
    if not 1 <= len(template) <= MAX_SNAPSHOT or len(template) != raw["template_length"]:
        raise ValueError("Invalid Ground template size")
    if codec.offset+codec.width > len(template) or codec.blank(template) != template:
        raise ValueError("Template does not mark a blank interpreted span")
    modulus,a,b,x0=(raw.get(k) for k in ("modulus","a","b","initial_state"))
    if (type(modulus) is not int or type(a) is not int or type(b) is not int or
            type(x0) is not int or not 2 <= modulus <= codec.capacity or
            not 0 <= a < modulus or not 0 <= b < modulus or
            not 0 <= x0 < modulus):
        raise ValueError("Invalid Ground transition description")
    return codec, template


@dataclass(frozen=True)
class TypedRelation:
    name: str
    input_type: str
    output_type: str
    work: int
    operation: str


class ComposedGroundNavigator:
    """Small typed graph: interpret -> virtual jump -> render -> digest."""

    def __init__(self, ground):
        self.ground=ground
        self.codec,self.template=_load_ground(ground)
        self.relations=(
            TypedRelation("interpret-span","observed.bytes","initial.scalar",1,"decode"),
            TypedRelation("navigate-affine","initial.scalar","terminal.scalar",2,"jump"),
            TypedRelation("materialize-span","terminal.scalar","realized.bytes",1,"render"),
            TypedRelation("digest-realization","realized.bytes","result.digest",1,"hash"),
        )
        # Retained powers are an optimization over successive intentions.
        self._powers=[(ground["a"],ground["b"])]
        self._powers_extended=0

    def route(self, result_type, max_work=10, max_hops=MAX_HOPS):
        targets={"state.scalar":"terminal.scalar",
                 "bytes.snapshot":"realized.bytes",
                 "digest.sha256":"result.digest"}
        if result_type not in targets:
            return {"status":"UNRESOLVED","reason":"UNSUPPORTED_RESULT_TYPE"}
        if (type(max_work) is not int or not 1 <= max_work <= 10000 or
                type(max_hops) is not int or not 1 <= max_hops <= MAX_HOPS):
            return {"status":"REFUSED","reason":"INVALID_COST_OR_HOP_CEILING"}
        target=targets[result_type]
        paths=[(0,0,(), "observed.bytes")]
        seen=set()
        while paths:
            cost,hops,names,state=heapq.heappop(paths)
            if state == target:
                stages=[next(r for r in self.relations if r.name==n) for n in names]
                return {"status":"QUALIFIED_MULTI_HOP","estimated_work":cost,
                        "stages":[{"name":r.name,"from":r.input_type,
                                   "to":r.output_type,"work":r.work} for r in stages]}
            if (state,hops) in seen or hops >= max_hops:
                continue
            seen.add((state,hops))
            for relation in self.relations:
                if relation.input_type==state and cost+relation.work<=max_work:
                    heapq.heappush(paths,(cost+relation.work,hops+1,
                                          names+(relation.name,),relation.output_type))
        return {"status":"UNRESOLVED","reason":"NO_ROUTE_WITHIN_CEILINGS"}

    def _powered_jump(self, x, steps):
        a,b,m=self.ground["a"],self.ground["b"],self.ground["modulus"]
        result_a,result_b=1,0
        index=0
        remaining=steps
        while remaining:
            if index==len(self._powers):
                pa,pb=self._powers[-1]
                self._powers.append((pa*pa%m,(pa*pb+pb)%m))
                self._powers_extended += 1
            power_a,power_b=self._powers[index]
            if remaining & 1:
                result_a,result_b = (power_a*result_a)%m,(power_a*result_b+power_b)%m
            remaining >>= 1
            index += 1
        return (result_a*x+result_b)%m

    def realize(self, steps, result_type="bytes.snapshot", *,
                max_work=10, max_hops=MAX_HOPS, include_data=False):
        if type(steps) is not int or not 0 <= steps <= MAX_STEPS:
            return {"status":"REFUSED","reason":"INVALID_NAVIGATION_HORIZON"}
        selected=self.route(result_type,max_work,max_hops)
        if selected["status"]!="QUALIFIED_MULTI_HOP":
            return selected
        observed=(self.template[:self.codec.offset] +
                  self.codec.encode(self.ground["initial_state"]) +
                  self.template[self.codec.offset+self.codec.width:])
        value=observed
        receipts=[]
        for stage in selected["stages"]:
            input_digest=domain_digest("vl64.composed-stage-input.v1",
                value if isinstance(value,int) else value)
            if stage["name"]=="interpret-span":
                value=self.codec.decode(value)
            elif stage["name"]=="navigate-affine":
                value=self._powered_jump(value,steps)
            elif stage["name"]=="materialize-span":
                value=(self.template[:self.codec.offset] +
                       self.codec.encode(value) +
                       self.template[self.codec.offset+self.codec.width:])
            elif stage["name"]=="digest-realization":
                value=_digest(value)
            else:
                raise ValueError("Unregistered Ground stage")
            receipts.append({"relation":stage["name"],"from":stage["from"],
                             "to":stage["to"],"input_digest":input_digest,
                             "output_digest":domain_digest(
                                 "vl64.composed-stage-output.v1",value)})
        result=({"type":result_type,"value":value} if isinstance(value,int) else
                {"type":result_type,"sha256":_digest(value),"length":len(value)}
                if isinstance(value,bytes) else
                {"type":result_type,"value":value})
        if isinstance(value,bytes) and include_data and len(value)<=4096:
            result["data_b64"]=base64.b64encode(value).decode("ascii")
        record={
            "status":"OBSERVED_GROUND_NAVIGATION",
            "ground_digest":self.ground["ground_digest"],
            "horizon":steps,
            "route":selected,
            "stage_receipts":receipts,
            "result":result,
            "power_basis_entries":len(self._powers),
            "interpretation_scope":"witnessed_rule_projection_not_proven_future_system_behavior",
        }
        record["receipt_digest"]=domain_digest("vl64.composed-ground-receipt.v1",record)
        return record

    def check_against_core(self, steps):
        """Independent core's dual-matrix/monoid derivation for a chosen horizon."""
        return self._powered_jump(self.ground["initial_state"],steps)==affine_navigate(
            self.ground["a"],self.ground["b"],self.ground["initial_state"],
            steps,self.ground["modulus"])
