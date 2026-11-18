# Non-Affine Virtual Grounds and Observable-State Compression

**Status:** Implemented and exercised in GitHub Actions on November 18, 2026.  
**Core:** 65,093 bytes of a 65,536-byte budget; this work changes **no** `src/vld/` source file.  
**Modules:** [discrete_ground.py](../experiments/byte_fabric/discrete_ground.py), [ground_families.py](../experiments/byte_fabric/ground_families.py), [quotient_ground.py](../experiments/byte_fabric/quotient_ground.py).  
**Evidence:** [non-affine Ground](../experiments/byte_fabric/datasets/2026-11-18-nonaffine-ground.json), [observable quotient](../experiments/byte_fabric/datasets/2026-11-18-observable-quotient-ground.json), and [measurements](../experiments/byte_fabric/datasets/2026-11-18-nonaffine-and-observable-measurements.json).

## The result

VL-64 is no longer forced to assume that a changing byte-state system follows a modular affine recurrence. The external model catalogue tries independently admitted descriptions and selects a candidate **only after it reconstructs held-out observations**. The chosen objective is the smallest JSON description among qualified candidates; this is a *finite registered-model comparison*, not an unbounded search for truth.

Two additional representations have been implemented:

1. A **closed finite-state Ground**: one fixed byte template, one automatically identified changing byte span, a finite preperiod, and a witnessed deterministic cycle. The compact runtime's `ObservedGround` cross-checks the transition graph. After loading the Ground, any modeled horizon can be navigated by a single cycle-index calculation, without replaying the preceding transitions.
2. An **observable quotient Ground**: a smaller state machine whose classes preserve an observer-specified output and all future outputs under the declared deterministic transition. It may discard exact internal byte identity because the observer's question does not require it.

An unknown source of bytes is still an exact byte Ground, without any requirement that one of these learned descriptions fit. Inference failure is **unresolved**, not a fabricated model.

## Discover a non-affine Ground from bytes

A trainer receives successive equal-length byte snapshots and separate held-out snapshots. No offset, token sequence or transition rule is supplied by the caller. The discovery pass:

- identifies a *single contiguous region* that changed across training frames and checks that all other bytes remained constant;
- treats the variable span as raw byte tokens, without assuming numeric endianness, an ISA, a mathematical law, or an interpreted programming language;
- builds witnessed `source → successor` relations and refuses conflicting successors;
- requires at least two complete witnessed traversals of a closed finite cycle;
- asks the existing compact `ObservedGround` implementation to verify deterministic successors and long-horizon navigation;
- reconstructs every training and held-out frame *exactly* before admitting the description.

The resulting content-addressed Ground records the byte template once, the preperiod token sequence, the cycle token sequence, evidence digests, and explicit unproven-future claims.

### Measured controlled experiment

The test machine emits 1,024-byte frames whose constant content is sampled from VL-64's checked-out README. Its **dynamics are independently generated synthetic state transitions**, not a claim that the README or a public service itself changes in this manner.

| Measurement | Result |
| --- | ---: |
| Training frames | 58 |
| Held-out frames | 52 |
| Additional unseen exact-frame checks | 4 |
| Original observed stream | **112,640 bytes** |
| Compressed finite-state Ground | **2,155 bytes** |
| Exact-model description ratio | **52.269×** |
| Automatically discovered variable field | 2 bytes at offset 427 |
| Prefix / closed cycle | 3 / 17 states |
| Modeled horizon | **1,000,000,000,000 steps** |
| Core runtime source bytes added | **0** |

A deliberately nonlinear transition mapping makes the modular-affine family fail admission. The competing closed-finite-cycle family passes. VL-64 navigates the trillion-step horizon by directly selecting the appropriate stored cycle token, reconstructs the frame and returns its SHA-256 digest, with stage receipts.

A separate 300,000-transition direct reference comparison measures the navigation improvement. This comparison uses synthetic controlled state changes and individual GitHub Actions runners. The result is **not** a general hardware speedup or proof of future external behavior.

### The observer speaks the same protocol

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/nonaffine_trillion.json \
  --out finite-state-result.json
~~~

This uses `ground.navigate` exactly as the earlier affine example. The adapter chooses the registered Ground driver from a validated schema; the observer chooses the output result and horizon rather than selecting the mathematical implementation.

## Compress for the observation, not for all possible internals

A second experiment starts with twelve finite internal states and observable outputs `A` and `B`. Merely grouping all current `A` states would be incorrect: some lead to `A` next, while others lead to `B` next.

VL-64 applies its existing **Myhill–Nerode-style partition refinement** over the completely declared, deterministic model. States are merged only if their current observable outputs and the successor **equivalence classes** agree. The partition is repeatedly refined until it is stable.

The quotient discovers **three** transition-stable observable classes rather than the naïve two classes obtained from current outputs alone. The quotient is independent of the exact byte tokens and the original full frame template.

| Observable measurement | Result |
| --- | ---: |
| Source states | 12 |
| Current output symbols | 2 (`A`, `B`) |
| Future-stable quotient states | **3** |
| States omitted by the quotient partition | 9 |
| Original observed source frames | 35,840 bytes |
| Standalone observable Ground | **743 bytes** |
| Newly checked observable continuations | 26 |
| Modeled horizon | **1,000,000,000,000 steps** |

These are two distinct kinds of compression:

- The finite-state Ground **preserves exact reconstructed bytes within the validated model**.
- The quotient Ground **preserves only the declared observable future**. It explicitly refuses requests for exact internal byte reconstruction, even if its output partition contains singleton states; the quotient-only artifact does not contain the original tokens.

This is *intention-conditioned computational compression*: the description retains the relationships required to answer the observer, rather than storing every state representation indiscriminately.

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/observable_quotient_trillion.json \
  --out observable-result.json
~~~

The portable 743-byte Ground can answer `state.observable` for modeled times without reloading the source byte history.

## Evidence and counterexamples

Both models can be challenged with additional observations:

- A full discrete Ground checks the actual bytes at a stated time index and records their digest if they disagree with the modeled reconstruction.
- An observable quotient checks the **external observer's labeled output** at an index, not its discarded bytes, and records the contradiction.

Every challenge is explicitly scoped to the supplied observations. A disagreement does not rewrite the original Ground, hide invalid predictions or authorize a different model. The recorded counterexample may be used to train and validate a *new* versioned Ground, while retaining the earlier version and the evidence that falsified it.

## Relation to established methods

The implementation draws on methods established in the automata-learning and state-abstraction literature:

- [Learning Moore Machines from Input-Output Traces](https://arxiv.org/abs/1605.07805) distinguishes observations, learned deterministic models and finite input/output traces.
- [Automata Learning with an Incomplete Teacher](https://www.cs.cornell.edu/~jnfoster/papers/lstar-with-blanks.pdf) motivates explicit unknowns rather than assuming unobserved behavior is known.
- [Scalable Learning of One-Counter Automata via State-Merging Algorithms](https://drops.dagstuhl.de/storage/00lipics/lipics-vol360-fsttcs2025/html/LIPIcs.FSTTCS.2025.35/LIPIcs.FSTTCS.2025.35.html) discusses passive state merging and its consistency obligations.

VL-64 implements a **restricted, directly witnessed deterministic orbit learner and finite observational quotient**, not Angluin's full active `L*` algorithm, RPNI or a general Mealy machine learner. The central safeguard is shared with those approaches: observations define a hypothesis, and a counterexample is evidence against that hypothesis.

## What to build next

1. **Input-conditioned Grounds.** The present finite learner has exactly one unlabeled `advance` action. Register multiple typed actions and distinguish states with action sequences; otherwise a remote API's branching behavior cannot be inferred.
2. **Sparse variable regions.** Permit several interacting spans with separately checked reconstruction maps and dependency edges. Do not erase unmodeled bytes merely because they appear noisy.
3. **Counterexample-guided revision.** Preserve the old Ground and its witnesses while learning a new version from contradictory observations. Avoid silently changing the historical meaning of a digest.
4. **Calibrated description selection.** Compare compact model size, capture bytes, transition prediction accuracy, proof obligations, parse costs and repeated query costs under one explicit objective.
5. **Remote interactive observations.** Use permitted network endpoints as source/action observers. Capture request and response bytes and test whether a stable model can predict new behavior under repeated requests. Do not infer closed dynamics merely from one successful fetch.
6. **Cross-host conformance.** Run the same saved Ground and intention on distinct observer-hosts; compare observable output and per-stage receipts, then independently fetch real data to challenge the model.

The ambition is to keep the **64 KiB interpreter interface fixed while its registered Grounds grow more expressive**. An internally navigable, compressed description is valuable only when the observer's desired result remains correct under real evidence.
