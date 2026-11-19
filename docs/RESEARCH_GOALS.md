# VL-64 Research Goals

## Purpose

Develop VL-64 as a compact, usable Virtual Layer utility. Each research path should result in a callable operation, a reproducible test, and an observable benefit to the runtime. The core package stays within a **65,536-byte source budget**; benchmarks, fixtures, adapters, and research reports can live outside that core.

This document is for intended work. The [README](../README.md) and [usage guide](USAGE_GUIDE.md) describe the interfaces available in the code today.

The step-by-step programme for connecting physical machine observations, software execution, screen buffers, hardware counters, and communication streams to VL-64 is in the [Machine Observation and Ground Discovery Plan](MACHINE_OBSERVATION_AND_GROUND_DISCOVERY_PLAN.md) and the [Virtual Internet Runtime Plan](VIRTUAL_INTERNET_RUNTIME_PLAN.md) (including a proposed separate network-native 64 KiB implementation).

The observer-facing input language, metadata-only model discovery experiment, output-format selection and next steps toward qualified remote execution are documented in [Observer Intention and Model Grounds](OBSERVER_INTENTION_AND_MODEL_GROUNDS.md).

For the implemented architecture-independent byte observer, extensible source and decoder registries, and the separation between observation and qualified execution, see [Universal Byte Ground and Execution](UNIVERSAL_BYTE_GROUND_AND_EXECUTION.md). The generic byte-fabric experiments live outside the 64 KiB core.

The first observer-specified result selector is implemented in [Goal-Directed Byte Navigation](GOAL_DIRECTED_BYTE_NAVIGATION.md). A typed goal is matched to admitted source/decoder/realizer relations rather than a fixed operator name.

[Compressive Virtual Grounds](COMPRESSIVE_VIRTUAL_GROUNDS.md) now demonstrates automatic byte-span interpreter discovery, empirically validated modular transitions, composable state navigation, exact frame reconstruction, and independent contradiction witnesses. The new code is external to the 64 KiB runtime core; the structural description remains a small, portable artifact.

The implemented [Non-Affine and Observable Grounds](NONAFFINE_AND_OBSERVABLE_GROUNDS.md) compare validated algebraic and finite-state descriptions, discover a closed non-affine state graph from bytes and refine the finite model into a smaller future-equivalent observer-specific quotient. They retain counterexample receipts while leaving the 64 KiB core unchanged.

[Active Action-Conditioned Grounds](ACTIVE_ACTION_GROUNDS.md) now examines the next research step: plan qualified observation sequences that distinguish competing finite state hypotheses, update the candidate set using actual byte evidence, preserve counterexamples and navigate the surviving modeled transition without iterative actuation. This first version-space prototype is explicitly not general L* learning.

[Trace-Derived Action Grounds](TRACE_DERIVED_ACTION_GROUNDS.md) extends active observation from testing host-provided candidates to **synthesizing the candidates directly from action/output bytes**, preserving unknown transitions as hypotheses, refusing inconsistent repeated observations, and resynthesizing from newly observed counterexamples. The experiment is completely external to the 64 KiB core.

[Monolithic Seed and Self-Grounds](MONOLITHIC_SEED_AND_SELF_GROUNDS.md) now demonstrates an exact, transportable, sub-64 KiB executable Python carrier with a source-derived Ground of Grounds, a verifying bootstrap, typed module import relations and deterministic reconstruction. The code remains outside the 64 KiB runtime. Its execution still requires an admitted Python-capable observer-host.

[Multi-Field and Partial-Observation Grounds](MULTIFIELD_PARTIAL_GROUNDS.md) extends the byte interpreter to several disjoint changing spans, joint action-conditioned states, known/unknown transitions and information-directed observation of only the missing bytes. It integrates through the external observer protocol without increasing the 64 KiB runtime.

## Development discipline

A proposed capability enters the runtime only when it has:

1. **An operation:** a defined input, output, error behavior, and public interface.
2. **A Ground:** an explicit state representation, transition rule, and interpretation of its data.
3. **Evidence:** an independent reference, invariant, trace, receipt, or counterexample that makes the result inspectable.
4. **A measurement:** time, work, memory, bytes, or another directly measurable cost against a comparison case.
5. **A compact implementation:** source-size measurement and a test showing the change still fits the 64 KiB package.
6. **A usage example:** a command or importable snippet a user can execute.

Preserve compatibility of established interfaces where possible. Keep transport adapters and large experiment harnesses separate from the small algebraic core. Report measured results distinctly from mathematical expectations.

## Translation research: from observed machines to predictive Grounds

The first callable translation path is `ObservedGround`: exact byte observation, declared byte projections, empirical transitions, held-out view comparison, information profiles, and cycle navigation with a Ground digest. This provides a measurable starting point for lifting software or hardware observations into virtual state.

The next research target is **description discovery**, not merely accepting declared offsets:

- **Acquire:** external, permissioned adapters produce raw snapshots and transition traces from files, processes, hardware registers, network responses, or computer-vision frame buffers. The 64 KiB core accepts bytes; source-specific capture belongs in adapters.
- **Describe:** discover candidate typed fields, operations, dependency relations, encodings, and predictive quotients from sampled transitions. Preserve the raw-byte mapping and representation identity.
- **Evaluate:** distinguish number of distinguishable states, reachable states, predictive coverage on unseen transitions, branching uncertainty, and cost of navigation. Maximize *validated, useful predictive states*, not the sheer number of possible byte strings.
- **Account for information:** measure empirical source/target entropy, conditional uncertainty and information gained or lost under an observation or projection. Test when a rule is reversible, many-to-one, or stochastic. Avoid conflating observed entropy changes with physical information creation.
- **Navigate and refine:** retain verified transition relationships, select reusable composed paths, compare outcomes with independent observations, and admit new descriptions only when they improve held-out prediction or measured realization cost.
- **Reconstruct:** require explicit data and verified inverse maps to reconstruct full original states; never fill unobserved fields by guessing. Compare multiple descriptions that preserve the same relevant observable.

**Acceptance:** one captured system supplies byte traces; two or more candidate representations are evaluated on held-out transitions; the chosen Ground predicts and navigates new supported states, preserves a checkable observation mapping, and demonstrably improves either prediction coverage or measured cost over its raw-byte baseline.

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
