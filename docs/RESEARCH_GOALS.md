# VL-64 Research Goals

## Purpose

Develop VL-64 as a compact, usable Virtual Layer utility. Each research path should result in a callable operation, a reproducible test, and an observable benefit to the runtime. The core package stays within a **65,536-byte source budget**; benchmarks, fixtures, adapters, and research reports can live outside that core.

This document is for intended work. The [README](../README.md) and [usage guide](USAGE_GUIDE.md) describe the interfaces available in the code today.

## Development discipline

A proposed capability enters the runtime only when it has:

1. **An operation:** a defined input, output, error behavior, and public interface.
2. **A Ground:** an explicit state representation, transition rule, and interpretation of its data.
3. **Evidence:** an independent reference, invariant, trace, receipt, or counterexample that makes the result inspectable.
4. **A measurement:** time, work, memory, bytes, or another directly measurable cost against a comparison case.
5. **A compact implementation:** source-size measurement and a test showing the change still fits the 64 KiB package.
6. **A usage example:** a command or importable snippet a user can execute.

Preserve compatibility of established interfaces where possible. Keep transport adapters and large experiment harnesses separate from the small algebraic core. Report measured results distinctly from mathematical expectations.

## Research tracks

### 1. General state-space navigation

**Goal:** Extend the existing affine jump and monoid composition into a reusable way to navigate other finite and structured state spaces.

- Define a small operator protocol: state, transition, composition, identity, horizon, and result.
- Add navigation for one additional class, such as linear recurrences or finite-state transition tables.
- Reuse quotienting and equivalence checks to avoid repeated work when states have the same future observable behavior.
- Emit a receipt describing the selected navigation method and verified terminal state.

**Acceptance:** for seeded small cases, jumped results equal independent step-by-step references; the cost of large horizons is recorded; invalid algebraic contracts fail an explicit test.

### 2. Ground descriptions, lenses, and observation

**Goal:** Make registered Grounds executable descriptions rather than passive registry entries.

- Establish versioned descriptions for state domains, operations, invariants, and observation projections.
- Connect `GroundRegistry` to actual lens and operator selection.
- Test byte-to-state-to-byte equivalence through `ByteLens` and record exact input/output digests.
- Use three-valued guards to distinguish confirmed, rejected, and unresolved observations.

**Acceptance:** a Ground can be registered, selected, executed, and checked through one public API; incompatible descriptions produce a reproducible non-admission result.

### 3. Evidence-carrying execution and recovery

**Goal:** Provide reusable receipts for transitions and support reconstruction after a process restart.

- Define a minimal receipt format binding intention, Ground, inputs, output, and verification evidence.
- Add a serializable journal/checkpoint format for `HashChainedJournal`, `PageSpace`, and effect records.
- Separate an intended external effect from its reported outcome; test interruption before and after effect execution.
- Validate lineage and authority without treating a received payload as authorization to execute it.

**Acceptance:** terminate and restart an independent process; reconstruct the committed state from saved evidence; demonstrate that ambiguous external-effect outcomes are reported distinctly from known commits.

### 4. Observer-host execution across communication paths

**Goal:** Make existing communication environments usable as qualified places to realize or observe an operation.

- Describe adapters by their accepted input, output, witness, cost, and required authority.
- Start with a local stream and one externally mediated path, keeping the same operation contract.
- Use `GitWireV2Parser` and `StreamPathMachine` as components in experiments where the path's own behavior produces a verifiable result.
- Verify the same mathematical intention across changes in transport or observer context.

**Acceptance:** show an operation realized by two distinct paths with equivalent checked results, authentic evidence of the path used, and measured transfer/compute costs. Do not count a forwarded claim as computed work without a transformation witness.

### 5. Structured tensor and attention evaluation

**Goal:** Turn structural numerical primitives into practical, benchmarked computational paths.

- Extend `StructuralLinearOperator` composition and factor handling for larger matrix families.
- Exercise FP8, FP4, block scales, normalization, and sparse expert routing on reference fixtures.
- Compose `FlashStreamState` accumulators from separate shards and verify the normalized result.
- Explore compact procedures for representing repeated or generated tensor structure instead of allocating dense weights.

**Acceptance:** compare against independently calculated dense or higher-precision reference outputs; publish absolute/relative error, bytes represented, and time per operation across multiple sizes.

### 6. Model structure and procedural realization

**Goal:** Use model structure as a Ground that can be inspected, addressed, and evaluated in small verifiable pieces.

- Read supplied artifact manifests and tensor-header metadata into canonical descriptions.
- Define and test tokenizer, tensor, layer, and activation contracts using a small fixture model.
- Execute an isolated layer or bounded model through the available operator paths.
- Measure what structure can be reconstructed from descriptions and what must be read from stored parameters.

**Acceptance:** reproduce a reference model fragment's outputs within a declared tolerance and emit a manifest-bound receipt identifying the operations and data used.

### 7. Exact consistency, proof, and representation checks

**Goal:** Strengthen the ability to determine whether descriptions and observations agree.

- Expand rational and finite-field examples around `ExactFieldMatrix` and `solve_sheaf_cohomology`.
- Add independent witness verification for inconsistent observation cycles.
- Broaden sparse fault tests with known injected faults and competing explanations.
- Explore proof-trace adapters as external tools while retaining a small certificate checker in the runtime.

**Acceptance:** every accepted result has a checkable witness; inconsistent examples produce counterevidence; randomly generated small instances agree with independent reference solvers.

### 8. Adaptive cost selection and structural rewrites

**Goal:** Make plan selection depend on measured work and demonstrated equivalence.

- Benchmark the current `direct_step` and `monoid_jump` implementations and compare estimates to measurements.
- Use `EGraph` and `GroundRegistry` to register candidate rewrites with preconditions.
- Connect `AutopoieticDiscoveryLoop` opportunity records to experiments that can certify and then admit a new realization.
- Compare multiple qualified plans componentwise using `NavigationCostVector`.

**Acceptance:** a new rewrite must preserve outputs over its declared domain; a chosen plan must satisfy explicit cost ceilings; performance claims must have reproducible measurements.

## Experiment record format

Use a small record for each experiment:

```text
ID:
Question:
Ground / operator:
Baseline:
Method:
Witness / independent check:
Measurements (environment, input size, time, bytes, error):
Outcome:
Next action:
```

Keep findings that survive checks in tests and callable utilities. Keep alternative hypotheses, negative controls, and larger experimental material in research records rather than expanding the README.

## Priority sequence

Start with **Ground execution and receipts** (tracks 2–3), then **a second navigable state family** (track 1). These provide a common contract for **path execution** and **tensor/model work** (tracks 4–6). Develop **consistency checking and cost-calibrated rewrites** (tracks 7–8) alongside each tested integration.

The benchmark for progress is straightforward: a new user can express an operation, execute it, inspect why the result was admitted, and repeat it on another valid representation or host while retaining the same mathematical meaning.
