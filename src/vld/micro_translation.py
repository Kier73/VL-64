"""Lift observed byte states into candidate Ground views and measured transitions.

Views select explicitly declared bytes of a snapshot. Inferred transitions
describe samples, not guaranteed behavior of an unobserved machine.
"""
from __future__ import annotations
import math
from collections import Counter
from itertools import combinations
from typing import Mapping, Optional, Sequence

from .micro_kernel import domain_digest


def _entropy(counts):
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n) if total else 0.0


class ObservedGround:
    """Learn finite transition relations over declared byte projections.

    `views` maps a name to byte offsets, or None to keep all bytes.
    `fit` takes observed (before, after) snapshot pairs; `evaluate`
    scores candidate views on separate held-out pairs.
    """

    def __init__(self, views: Optional[Mapping[str, Optional[Sequence[int]]]] = None):
        source = {"bytes": None} if views is None else views
        if not source:
            raise ValueError("At least one view is required")
        self.views = {}
        for name, offsets in source.items():
            if not isinstance(name, str) or not name:
                raise ValueError("View names must be nonempty strings")
            if offsets is None:
                self.views[name] = None
            else:
                indices = tuple(offsets)
                if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
                    raise ValueError("Offsets must be distinct nonnegative integers")
                self.views[name] = indices
        self.edges = {name: {} for name in self.views}

    def _key(self, raw: bytes, view: str) -> str:
        if not isinstance(raw, bytes):
            raise TypeError("Snapshot must be bytes")
        offsets = self.views[view]
        if offsets is None:
            return raw.hex()
        if any(i >= len(raw) for i in offsets):
            raise ValueError("Snapshot shorter than declared view")
        return bytes(raw[i] for i in offsets).hex()

    def lift(self, raw: bytes) -> dict:
        """Return a digest-bound byte observation and projected state values."""
        return {
            "byte_digest": domain_digest("vl64.observation.v1", raw),
            "byte_count": len(raw),
            "states": {name: self._key(raw, name) for name in self.views},
        }

    def fit(self, pairs) -> "ObservedGround":
        """Accumulate witnessed transitions; samples may contain repeated pairs."""
        for before, after in pairs:
            # Validate every projection before modifying this pair's counts.
            keys = [(name, self._key(before, name), self._key(after, name))
                    for name in self.views]
            for name, src, dst in keys:
                self.edges[name].setdefault(src, Counter())[dst] += 1
        return self

    def fingerprint(self) -> str:
        """Content identity of view definitions and observed transition counts."""
        description = {
            "views": self.views,
            "edges": {name: {src: dict(sorted(counts.items()))
                             for src, counts in sorted(edges.items())}
                      for name, edges in sorted(self.edges.items())}
        }
        return domain_digest("vl64.ground.v1", description)

    def predict(self, raw: bytes, view: str) -> dict:
        """List observed successors and empirical branching of a state."""
        state = self._key(raw, view)
        counts = self.edges[view].get(state, {})
        total = sum(counts.values())
        return {
            "view": view, "state": state,
            "status": ("UNOBSERVED" if not total else
                       "SINGLE_OBSERVED_SUCCESSOR" if len(counts) == 1 else
                       "MULTIPLE_OBSERVED_SUCCESSORS"),
            "observations": total,
            "outcomes": dict(sorted(counts.items())),
            "conditional_entropy_bits": _entropy(counts),
        }

    def profile(self, view: str) -> dict:
        """Empirical state counts and information measures over fitted pairs."""
        edges = self.edges[view]
        src = Counter({k: sum(v.values()) for k, v in edges.items()})
        dst = Counter()
        for targets in edges.values():
            dst.update(targets)
        samples = sum(src.values())
        conditional = (sum(n * _entropy(edges[k]) for k, n in src.items()) / samples
                       if samples else 0.0)
        source_h, target_h = _entropy(src), _entropy(dst)
        return {
            "view": view,
            "pairs": samples,
            "source_states": len(src),
            "target_states": len(dst),
            "observed_states": len(set(src) | set(dst)),
            "single_successor_sources": sum(len(v) == 1 for v in edges.values()),
            "multiple_successor_sources": sum(len(v) > 1 for v in edges.values()),
            "source_entropy_bits": source_h,
            "target_entropy_bits": target_h,
            "conditional_entropy_bits": conditional,
            "predictive_information_bits": max(0.0, target_h - conditional),
            "entropy_change_bits": target_h - source_h,
        }

    def evaluate(self, pairs) -> list:
        """Rank views using independent held-out transitions.

        Score = target entropy * correct most-likely predictions / total pairs.
        This is a practical selection heuristic, not a formal capacity proof.
        """
        outcomes = {name: {"src": [], "dst": [], "correct": 0, "covered": 0}
                    for name in self.views}
        total = 0
        for before, after in pairs:
            keys = [(name, self._key(before, name), self._key(after, name))
                    for name in self.views]
            total += 1
            for name, src, dst in keys:
                record = outcomes[name]
                record["dst"].append(dst)
                record["src"].append(src)
                counts = self.edges[name].get(src)
                if counts:
                    record["covered"] += 1
                    # Count-based guess, not an oracle for ambiguous successors.
                    chosen = sorted(counts, key=lambda k: (-counts[k], k))[0]
                    record["correct"] += (chosen == dst)
        if not total:
            raise ValueError("At least one validation pair is required")
        scores = []
        for name, record in outcomes.items():
            target_entropy = _entropy(Counter(record["dst"]))
            correctness = record["correct"] / total
            scores.append({
                "view": name,
                "sample_count": total,
                "coverage": record["covered"] / total,
                "accuracy": correctness,
                "target_entropy_bits": target_entropy,
                "score": target_entropy * correctness,
                "distinct_targets": len(set(record["dst"])),
            })
        return sorted(scores, key=lambda s: (-s["score"], -s["coverage"],
                                             -s["distinct_targets"], s["view"]))

    @classmethod
    def discover(cls, training, validation, max_positions: int = 8,
                 max_fields: int = 2) -> tuple:
        """Try small byte-coordinate views; rank on independent observations.

        Search is bounded to the first max_positions bytes and at most
        max_fields selected positions per view.
        """
        if type(max_positions) is not int or type(max_fields) is not int or max_positions < 1 or max_fields < 1:
            raise ValueError("Search dimensions must be positive integers")
        training, validation = list(training), list(validation)
        if not training or not validation:
            raise ValueError("Training and validation transitions are required")
        length = min(len(raw) for pair in training + validation for raw in pair)
        width = min(length, max_positions)
        if not width:
            raise ValueError("No byte positions available")
        views = {"bytes": None}
        for k in range(1, min(max_fields, width) + 1):
            for offsets in combinations(range(width), k):
                views["byte_" + "_".join(map(str, offsets))] = offsets
        ground = cls(views).fit(training)
        return ground, ground.evaluate(validation)

    def navigate(self, raw: bytes, steps: int, view: str) -> dict:
        """Jump through an observed deterministic transition graph via cycle skip.

        Returns a projected state, not reconstructed full source bytes.
        A multi-successor or unobserved state halts rather than guessing.
        """
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        state = self._key(raw, view)
        initial = state
        seen = {}
        sequence = []
        evaluated = 0
        cycle = 0
        steps_skipped = 0
        while evaluated < steps:
            if state in seen:
                start = seen[state]
                cycle = evaluated - start
                state = sequence[start + (steps - start) % cycle]
                steps_skipped = steps - evaluated
                break
            targets = self.edges[view].get(state)
            if not targets or len(targets) != 1:
                raise ValueError("Observed transition is missing or ambiguous at state " + state)
            seen[state] = evaluated
            sequence.append(state)
            state = next(iter(targets))
            evaluated += 1
        return {
            "status": "OBSERVED_MODEL_NAVIGATION",
            "view": view,
            "ground_digest": self.fingerprint(),
            "start_state": initial,
            "terminal_state": state,
            "steps": steps,
            "evaluated_steps": evaluated,
            "skipped_steps": steps_skipped,
            "cycle_length": cycle,
        }
