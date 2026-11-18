# Trace-Derived Action Grounds: Learning the Interpreter from Byte Evidence

**State:** Executable experimental implementation, November 18, 2026.  
**Compact runtime:** no edits to `src/vld/`, total **65,093 / 65,536 bytes**.  
**Code:** [trace_synthesis.py](../experiments/byte_fabric/trace_synthesis.py), [trace_lab.py](../experiments/byte_fabric/trace_lab.py), the existing [active_ground.py](../experiments/byte_fabric/active_ground.py) and [active_adapter.py](../experiments/byte_fabric/active_adapter.py).

## Why this is a new capability

In the prior active-Ground experiment, VL-64 could choose the most informative action *among preinstalled candidate machines*. The external observer still had to supply the transition tables.

Here **the external observer only needs an exact, bounded record of action words and returned bytes**. The new synthesizer constructs its own plausible finite deterministic Ground descriptions from those traces, then uses the existing active navigator to select further distinguishing experiments.

The intended mechanism is:

~~~text
resettable host / communication observation
  -> immutable action/output evidence
  -> observed byte-output prefix tree
  -> bounded deterministic state merging + hypothesis completion
  -> candidate Ground set (OBSERVED edges vs GUESSED edges)
  -> information-gain action planner
  -> qualified host probe returning actual bytes
  -> preserve contradiction / update evidence ledger
  -> resynthesize candidates
  -> held-out conformance testing
  -> qualified, internally navigable finite Ground
~~~

This genuinely moves **hypothesis generation** inside VL-64's external research system. It does not yet discover arbitrary programs, CPU instruction sets, transformer internals or remote network automata. Those require more expressive observation and transition abstractions.

## The evidence contract

A `TraceLedger` admits a declared, bounded action alphabet and complete byte responses for each requested action word. A word is executed from a **host-declared reset initial state**. The observation is a sequence of output byte strings, one for each action.

The recorded evidence retains the input word, exact output bytes as hex, each SHA-256 digest, the host source identifier, the declared authority policy, and a digest of the whole record.

**Unknown transitions remain unknown evidence.** They are not added to the ledger merely because a candidate machine predicts an output.

A repeated action prefix returning different bytes is a contradiction of the deterministic/resettable observation assumption. The ledger preserves that contradiction and the synthesizer refuses to issue a deterministic Ground from those conflicting records.

Saved ledgers are validated by replaying every recorded action-output receipt and its domain-separated digest. A saved ledger is immutable relative to subsequent observations: new experiments produce new evidence and a new ledger fingerprint, rather than altering the old snapshot.

These are integrity checks, not cryptographic proof that a claimed external host actually performed the actions.

## How candidate machines are synthesized

The first learner constructs a **Mealy-style prefix tree**: every distinct observed input prefix becomes a provisional state, and an edge exists only where an action/output response was actually witnessed.

It then considers small state quotients by assigning prefix-tree nodes to a bounded number of latent states. Assignments are rejected whenever they imply incompatible outputs or successor states for the same state/action pair.

Each consistent partial graph may be completed into a full candidate transition table. Importantly, unobserved transitions are **only conjectures**: their output symbols are drawn from outputs already observed, and the selected successor is one of the currently declared candidate states. This is a finite hypothesis-family assumption, not a discovery that all possible outputs have been observed.

Candidate enumeration is bounded by state count, observed prefix-tree nodes, output symbols, attempted assignments, and maximum candidate count. It returns whether search was truncated. It never asserts global uniqueness or that an unobserved edge is certified.

The `TraceGroundSynthesizer` uses the already-built `ActionHypothesis` class as its **candidate descriptor format**, preserving the existing authority and navigation contracts.

## Active observation cycle

Given at least two synthesized candidates, the existing `ActiveGroundSession` predicts how each will respond to possible short action words. It chooses the word maximizing expected distinguishing information per declared cost.

The selected word is run **only through a host-installed, qualified `QualifiedResetProbe`**. No JSON input is allowed to install an executor, trigger an unapproved remote mutation or run arbitrary host code.

The actual outputs are recorded, then candidate synthesis is performed again using the enlarged evidence. A newly discovered output symbol can contradict *all earlier guesses*; that is expected and does not make the new observation invalid. The system simply constructs new hypotheses consistent with the new evidence.

The observer may request this operation through the existing typed protocol:

~~~json
{
  "operation": "ground.traces.plan",
  "target": "workspace:experiments/byte_fabric/datasets/2026-11-18-initial-trace-ledger.json",
  "output": "json",
  "options": {
    "max_states": 2,
    "max_depth": 3,
    "max_words": 64
  }
}
~~~

This returns the synthesized candidate summary, their Ground digests, and the **proposed next action experiment**. The operation does *not* run that experiment or claim the candidate models have been independently validated.

The existing `ground.action.navigate` operation subsequently opens an admitted saved Ground and navigates repeated modeled transitions through a bounded power-composition algorithm.

## Measured controlled experiment

The reference host was a completely separate, resettable two-state machine that responds to `flip` and `read` actions. VL-64 did not receive its internal state count or transition rules as input to the synthesizer—only the action alphabet and a small initial set of observed action/output traces.

| Quantity | Measured result |
| --- | ---: |
| Initial observed action words | 2 |
| Automatically synthesized initial candidates | **14** |
| Additional active distinguishing probes | **3** |
| Final training traces, including additional coverage probes | 10 |
| Reference oracle calls | 10 |
| Total action steps observed in training | 25 |
| Exact reference output bytes returned | 68 |
| Final selected Ground states | **2** |
| Hypothesized/unobserved edges in selected Ground | **0** |
| Independent held-out action words checked | **62 / 62 matching** |
| Modeled jump horizon | **1,000,000,000,000 repeated flips** |
| Finite transformation power compositions | **53** |

The learner's three chosen active probes were `read → flip → read`, `flip → flip → read`, and `flip → read → read`. Additional action-word coverage was also gathered as **training evidence**, separately from the 62 held-out validation sequences.

A newly observed output (`one`) was absent from the initial byte-output alphabet, demonstrating why merely treating the first candidate set as complete would be incorrect. The synthesizer incorporated it and regenerated candidate models.

The final two-state Ground was chosen from **training evidence only**, then tested against every held-out action word of lengths one through five over the two-action alphabet. All 62 matched. This is exhaustive *for that fixed finite test suite*, not proof of equivalence over arbitrary-length words or larger hidden machines.

The model's trillion-step result is a **mathematical projection of the retained finite machine**. No trillion-step reference host run, network actuation or independent proof of remote future behavior is claimed.

For this tiny training dataset, the descriptive Ground plus its evidence may exceed the 68 raw response bytes. The benefit demonstrated is **reusable computational structure and fast navigation**, not an already established net compression ratio for the training trace. Compression should be assessed against total verified computation and repeated use, including learning and validation costs.

## Real Internet evidence

The optional live experiment observes the complete byte representations of two pinned public GitHub documents over HTTPS under a restricted `read-only-remote` host policy. Each actual action emits the SHA-256 bytes of an acquired document, and the synthesizer constructs candidate finite models from those **real response bytes**.

The recorded observations retain HTTP metadata, source URL, byte count, timing and digest. The next action word is selected by the same candidate-discriminating planner rather than hardcoded by the observer.

This is a valid Internet-computation path for source observation and byte interpretation. It is **not evidence** that reading static GitHub files changes the server into a learnable stateful machine; the experiment is explicitly classified as *stateless resource observation*. Remote stateful systems will require reset/replay authority and controlled interaction.

## Reproduce

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.byte_fabric.trace_lab --out artifacts

python -m experiments.observer_intent.arbitrate \
  --intent artifacts/trace-synthesis-intention.json \
  --out artifacts/suggested-next-action.json

python -m experiments.observer_intent.arbitrate \
  --intent artifacts/learned-action-navigation-intention.json \
  --out artifacts/modeled-state.json

# Optional; network availability may change:
python -m experiments.byte_fabric.trace_lab --out artifacts --live-only
~~~

The versioned Ground, raw trace evidence, measurements and two example observer intentions live in [experiments/byte_fabric/datasets](../experiments/byte_fabric/datasets/) and [experiments/byte_fabric/intents](../experiments/byte_fabric/intents/). GitHub Actions checks on Python 3.11 and 3.13 include the full earlier test suite and the 65,536-byte core ceiling.

## Relationship to established research

This implementation is a **bounded trace-prefix-tree / state-merging enumerator**, not a complete active `L*` or `L#` learning algorithm.

The separation between partial observed input/output trees, inferred finite models, distinguishing queries and counterexamples draws on:

- [A New Approach for Active Automata Learning Based on Apartness](https://arxiv.org/abs/2107.05419) — constructive distinctions over observation trees.
- [Automata Learning with an Incomplete Teacher](https://www.cs.cornell.edu/~jnfoster/papers/lstar-with-blanks.pdf) — unknown observations and incomplete evidence.
- [Active learning of symbolic Mealy automata](https://www.sciencedirect.com/science/article/pii/S0304397526004032) — the challenge of large input alphabets and typed outputs.
- The related [In-Formation-Space transition-congruence](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/observation-equivalence/transition_congruence.py) and [distinguishing-navigation](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/observation-equivalence/distinguishing_navigation.py) utilities.

### Next development

1. **A genuinely partial action Ground:** retain observed transition edges separately from completion hypotheses, then update only missing edges as new bytes arrive.
2. **New output symbols and variable structure:** permit expansion beyond previously seen output alphabets without requiring an entirely new candidate enumeration.
3. **Closure/consistency of learned state representations:** derive minimal observational state signatures from access sequences and distinguishing suffixes rather than searching arbitrary bounded merges.
4. **Qualified HTTP and process actions:** observe request/response frames, authorization, ordering and reset behavior; distinguish genuinely stateful endpoints from static resource reads.
5. **End-to-end measured compression:** optimize transferred bytes + probes + verification + stored description + navigation; report when learning costs dominate.
6. **Federated independent challenge:** send a Ground and planned action to another observer-host, compare separately acquired bytes and retain any disagreement.

The direction remains unchanged: **external intention → informative observation → mathematical Ground → navigable state description → independently tested result**, without expanding the 64 KiB core.
