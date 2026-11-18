# Compressive Virtual Grounds — Interpreter Discovery and Self-Navigating State

**Status:** executable controlled research, November 18, 2026.  
**Compact core:** unchanged; **65,093 bytes of 65,536**, with 443 bytes remaining.  
**Implementation:** [orbit_ground.py](../experiments/byte_fabric/orbit_ground.py), [orbit_adapter.py](../experiments/byte_fabric/orbit_adapter.py), and [orbit_lab.py](../experiments/byte_fabric/orbit_lab.py).  
**Evidence:** [learned Ground](../experiments/byte_fabric/datasets/2026-11-18-compressive-orbit-ground.json) and [cross-runtime measurements](../experiments/byte_fabric/datasets/2026-11-18-compressive-orbit-measurements.json).

## Core proposition

A Virtual Ground can represent more than a compressed archive of observed states. It can be a **compact description of how a family of states is generated**, allowing an observer to navigate to a requested state without materializing its predecessors.

In this implementation a fixed-layout byte snapshot is described by:

\[
G = (\text{constant byte template}, \text{interpreted variable span},
x_0, a, b, m, \text{observation evidence})
\]

with a candidate transition law:

\[
x_{t+1}=(a x_t+b)\bmod m
\]

and a reconstruction operator that inserts the computed state into its original byte template. The general byte-fabric remains the common external representation: any captured source begins with actual bytes, regardless of architecture. This particular **compressive interpreter** is admitted only when observations justify its assumptions.

**Interpretation is not automatic truth.** The observed finite sequence can validate an affine hypothesis on held-out frames; it cannot prove the physical or software source must follow the same law forever.

## What makes this a computational Ground

The stored description contains a single byte template, a typed variable lens, an initial state, two affine coefficients, a modulus, and digests of training and validation evidence. It does **not** retain the individual 80 observed snapshots.

Given an observer request for state index \(t\), VL-64 composes the following relations:

<escape>1.</escape> \`observed.bytes → initial.scalar\`: interpret the variable span using the learned codec.  
<escape>2.</escape> \`initial.scalar → terminal.scalar\`: apply a modular affine jump through retained powers of the transition.  
<escape>3.</escape> \`terminal.scalar → realized.bytes\`: reconstruct the byte snapshot from the static template plus selected scalar.  
<escape>4.</escape> \`realized.bytes → result.digest\`: compute the requested output's SHA-256.

The first three stages can return a snapshot; the first two can return a scalar. The fourth adds a digest without changing the observer's source language or asking the observer to write the algorithm.

The typed graph uses a **bounded, costed path search** rather than assuming a single operation. Every stage carries input/output type and input/output content-digest evidence. A separate \`ObserverIntention\` selects target kind, time horizon, hop ceiling, work ceiling and desired output format.

## Discovering an interpreter, not supplying its offsets

The discovery procedure inspects sequences of raw byte snapshots with equal lengths and looks for a **single contiguous region whose bytes changed** during training. It tries independent small interpretations—little-endian unsigned, big-endian unsigned and fixed-width decimal ASCII—fits possible modular affine transitions, and validates each candidate against held-out snapshots.

A decoder is admitted only if **exactly one** candidate matches every required observation. Multiple independent changing fields, an ambiguous modular coefficient, incompatible framing, or inconsistent transitions result in refusal.

The current discovery is deliberately a small, strong experiment in learning an interpreter from data. It is not a claim to automatically reverse-engineer every instruction set, file format, web page, graphics framebuffer or model weight layout.

Once selected, the interpreter and transition description are portable: another observer can receive the small Ground and navigate without receiving the full training stream. Its future state outputs are model-derived predictions until independently checked against the actual source.

## Executed experiment: a virtual orbit over bytes

A controlled process generated **1,024-byte frames** containing one changing four-byte state and a much larger constant payload. The experimental code did not supply the variable's offset or endianness to the interpreter discovery phase.

| Measurement | Observed result |
| --- | ---: |
| Training frames | 32 |
| Held-out validation frames | 48 |
| Additional unseen frames reconstructed exactly | 5 |
| Raw training + held-out snapshots | 81,920 bytes |
| Compressed Ground JSON description | 1,947 bytes |
| Source-to-Ground representation ratio | **42.075×** |
| Automatically discovered interpretation | four-byte unsigned little-endian at offset 431 |
| Learned transition coefficients | \`a=1664525\`, \`b=1013904223\`, modulus \`2^32\` |
| Navigated horizon | **1,000,000,000,000** transitions |
| Number of composed relations for requested digest | 4 |
| Core package source size | 65,093 bytes |

The Ground reconstituted all tested observed and held-out frames exactly. The unseen-frame checks used frames outside both the fitting and validation datasets. The distant state was independently cross-checked using the VL-64 core's existing two mathematical derivations of modular affine navigation.

In a separate **200,000-step** direct reference comparison, the measured speedup of the first compressed-state navigation was approximately **310× on the Python 3.11 runner** and **141× on Python 3.13**. Those are observations of this one machine/algorithm/workload; cold versus warm basis effects and operating-system scheduling differ between runs. They are not universal hardware speedups, and direct stepping for one trillion transitions was not performed.

Crucially, the saved Ground requires **no knowledge of the state generator's source code** to produce the tested reconstruction. It learns the transition description from observations, then navigates that admitted description.

## Reproduce the result

From a VL-64 checkout:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.byte_fabric.orbit_lab --out artifacts

python -m experiments.observer_intent.arbitrate \
  --intent artifacts/trillion-step-intention.json \
  --out artifacts/trillion-navigation.json
~~~

The [versioned Ground example](../experiments/byte_fabric/intents/ground_trillion.json) also works as an observer request independently of the data-generation step:

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/ground_trillion.json \
  --out navigated-ground-result.json
~~~

This reads the saved Ground description through the existing authorized workspace adapter, verifies the description's content digest, discovers the typed relation path, navigates the modeled horizon, then returns JSON. JSONL and text are also supported by changing the observer-selected \`output\`.

The receipt includes the original request, model identity, chosen path, per-stage input/output digests, the returned value or bytes, and the evidence scope. The **Ground digest is an integrity fingerprint, not authentication of the source or a proof of perpetual predictive accuracy**.

## Why this differs from byte-level compression

Traditional lossless compression maps a finite input message to a shorter encoding and reconstructs that same input. Here the Ground represents a **generator for a whole sequence of states**. Within the empirically validated law and fixed-template assumptions, it can reconstruct previously observed states and describe future model states whose raw snapshots were never stored.

This is a form of *computational description compression* or *model-based state compression*. Its compression ratio is measured against the retained observed snapshot stream—not against the total information content of arbitrary future source states. An incorrect learned law would produce incorrect predictions, so validation and refusal behavior are central to whether the compression is useful.

### What is actually conserved

The variable state remains explicit. The fixed context is stored once. The transition rule is stored once. For every navigation request, the state at time \(t\) is derived using composition of transition functions; no intermediate frames need to be replayed.

Under this model:

\[
\text{StateBytes}(t)=\text{Template}\big[\text{Span}\leftarrow T^t(x_0)\big]
\]

and \(T^t\) is computed with reusable powers of \(T\) in \(O(\log t)\), not by running \(t\) physical stepping iterations.

This gives the Ground an intrinsic *navigation operation*: the interpreter describes the state, and the state-space description itself provides the transition mechanism.

## Research progression

1. **Multiple variable regions.** Learn relational or dependency-preserving projections instead of assuming one changing scalar field; allow independently validated cross-field transitions and exact typed reconstruction.
2. **Non-affine finite state spaces.** Reuse the compact core's \`ObservedGround\` transition and cycle navigation to represent finite deterministic quotients, then compare their description cost to algebraic models.
3. **Different mathematical structures.** Consider modular vector recurrences, sparse tensor operators, recurrence-free algebraic solutions, conditional rewrite systems and composable partial observations; admit them only when their witnesses justify future navigation.
4. **General interpreter catalogue.** Host-installed decoders should advertise which byte representations, variable relationships and state laws they support; a Ground search should select the smallest sufficient *qualified* model.
5. **End-to-end cost.** Include acquisition, interpretation discovery, validation, storage, repeated queries, recovery and independent verification; an apparently smaller Ground is not automatically the cheaper overall computation.
6. **Independent observer-hosts.** Send the same saved Ground to a different host and compare modeled outputs to independently acquired source states. Keep source authenticity separate from mathematical agreement.
7. **Learning from contradiction.** When a new observation falsifies an assumed transition, mark the Ground invalid for that interval, retain the contradiction and search for a refined description—never silently overwrite evidence.

The long-term objective is an interpreter whose valid descriptions make new computational relationships navigable **without hardcoding each original architecture**. The immediate version demonstrates this under a precisely declared empirical model and retains the 64 KiB core.
