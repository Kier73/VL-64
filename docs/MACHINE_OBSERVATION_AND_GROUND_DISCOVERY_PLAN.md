# VL-64 — Physical Host Observation and Ground Discovery Plan

**State:** Research and experimental implementation plan, November 18, 2026.  
**Primary deliverable:** A reproducible, bottom-up machine-observation laboratory that proposes, evaluates, and exports navigable Grounds for the 64 KiB VL-64 runtime.  
**Documentation contract:** This plan describes intended work. The [README](../README.md) and [usage guide](USAGE_GUIDE.md) remain about implemented functions.

## Objective

Observe running software, memory representations, screen buffers, system hardware, and communication streams; identify typed variables and relations that explain their changes; construct candidate Virtual Grounds; choose the descriptions with the best independently measured predictive and computational properties; and realize selected intentions via verified, potentially faster navigation.

A byte string always admits an exact byte-vector description. A *useful* Ground additionally identifies a domain, the observer and action context, the state variables that matter, transition laws or hypotheses, invariants, and evidence connecting them back to observations.

Success is not the largest imaginable state count. It is **useful distinguishability and reachable-state coverage under an intention, combined with held-out predictive yield, verification strength, and measured execution cost**. A representation may collapse many byte-distinct states while preserving all observations needed for a particular task. That can be an improvement, not a loss.

Keep two complementary products:

1. **Host laboratory (outside the 64 KiB source budget):** capture, probes, parsers, image processing, broad candidate search, benchmarking, reporting, and optional accelerators.
2. **VL-64 core (inside the 65,536-byte package budget):** compact Ground descriptions and fingerprints, declared lenses, guarded state operations, compositional navigation, small verifiers, and receipts.

At plan creation, the audited VL-64 package has 63,528 source bytes, leaving 2,008 bytes. Do not reserve those bytes for an all-in-one camera, OCR, ETW, network, or hardware discovery implementation. Extract reusable primitives only after measurement and source-budget review.

## A. Existing capabilities to compose

This plan is informed by actual implementations found in related research repositories, not a mandate to transplant their code wholesale.

| Existing source | Actual useful method | Intended use in this plan |
| --- | --- | --- |
| VL-64 [ObservedGround](../src/vld/micro_translation.py) | Byte views, empirical successor counts, entropy profiles, held-out scoring, cycle indexing | Current narrow baseline and compact target interface |
| [DataGround / typed lenses](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/data_ground.py) | Byte identity plus explicitly selected, reversible candidate interpretations of UTF-8, integers, JSON, IPv4 data | Typed-decoder adapters with exact round-trip claims |
| [Active Ground discovery](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/active_ground_discovery.py) | Permissioned observation callbacks, finite witness fitting, withheld probes, subprocess timeouts | Live-process probe protocol and conservative inference |
| [Predictive description selection](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/docs/PREDICTIVE_DESCRIPTION_SELECTION.md) | Chronological partitions, coverage/yield/residuals, separate state support and cost, Pareto frontier | Replace single-score selection with multiobjective, chronological evidence |
| [Live capture provenance](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/live_router_capture.py) | Capture-time chains and checked raw packet evidence | Distinguish acquisition evidence from an after-the-fact record |
| [Binary/process lifting](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/substrates/lifting.py) | ELF64/WASM bounded lifting, content-addressed process-state operations | Binary formats and restricted executable-state adapters |
| [Active distinguishing probes](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/inquiry/probes.py) | Greedy/beam/minimax probe selection, explicit unresolved partial transitions | Choose observations that discriminate competing models |
| [Partial-aware attention](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/7a3c4a368d1d24044a6c06a317c26ef316513e6d/virtual_ground_interpreter/vgi/partial_attention.py) | Counterfactual probe scoring on incomplete Grounds | Allocate capture/probe effort to impactful missing cells |
| [Bounded program law extractor](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/7a3c4a368d1d24044a6c06a317c26ef316513e6d/virtual_ground_interpreter/vgi/program_law.py) | Safe AST-subset extraction of executable transition laws | Translate known source logic without executing arbitrary code as a parser |
| [Polynomial lifting](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/7a3c4a368d1d24044a6c06a317c26ef316513e6d/virtual_ground_interpreter/vgi/polynomial_lift.py) | Candidate finite-field polynomial descriptions with whole-supplied-table validation | Move beyond byte-offset transition lookup |
| [Observation equivalence](https://github.com/Kier73/In-Formation-Space/blob/48d8d3a7b8c8e2adce3fb6067a13b0227adb15da/bridges/observation-equivalence/distinguishing_navigation.py) | Find permitted future actions that distinguish apparently equivalent states | Learn sufficient predictive state, rather than preserving every observed byte |
| [Operation span recorder](https://github.com/Kier73/In-Formation-Space/blob/48d8d3a7b8c8e2adce3fb6067a13b0227adb15da/bridges/observation-path/operation_spans.py) | Synchronous timing receipts and cost profiles | Measure capture, translation, verification, navigation and direct execution separately |

**Transfer rule:** Prefer an adapter or research-only dependency over copying a whole subsystem. Reuse the mathematical contract and fixtures; introduce code into VL-64 only where a smaller independently tested implementation adds value.

## B. Target system interfaces

### B1. Capture envelope (external laboratory)

Every captured observation must retain enough information to reconstruct what was measured and to pair it with the correct action. Proposed JSON-lines manifest (large raw payloads remain local sidecar files):

~~~json
{
  "schema": "vl64.observation.v1",
  "session": "run-001",
  "source": "controlled-app/process-1",
  "kind": "process-state",
  "sequence": 42,
  "monotonic_ns": 1234567890123,
  "wall_utc": "2026-11-18T12:00:00Z",
  "clock_scope": "host-1/boot-1",
  "action_id": "step-42",
  "action_phase": "after",
  "encoding": "application/octet-stream",
  "byte_length": 64,
  "sha256": "<actual digest>",
  "blob": "blobs/<sha256>.bin",
  "capture_adapter": "process-json-v1",
  "authorization": "local-opt-in"
}
~~~

Fields must contain real values when an implementation emits a record. The example digest and timestamp are placeholders, not a captured event. Use content-addressed blobs and hash the actual bytes; scope monotonic timestamps to one clock domain and do not silently equate wall-clock time with event index. Record lost frames, dropped ETW events, counter reset, capture resolution, decode failures, and incomplete before/after pairs.

### B2. Observation adapter contract

Adapter operations:

- \`describe()\`: return source identity, OS/driver/producer version, units, capture permissions, and schema.
- \`capture(action_id=None)\`: emit an immutable observation envelope plus raw bytes or a reference.
- \`apply(action)\`: optional **authorized and explicitly scoped** action on a controlled test target, returning an action receipt; read-only adapters omit this.
- \`close()\`: release capture resources, flush receipts, and disclose dropped observations.

The adapter does not decide which mathematical model is true. Capture identity, data integrity, authorization to act, semantic interpretation, and proof of a law are distinct layers.

### B3. Candidate Ground contract

A proposed \`GroundDescription\` should minimally carry:

- \`ground_id\`, \`schema_version\`, source-observation digests, and a versioned description digest;
- typed state variables with path/offset/ROI, encoding, units, and exact/approximate extraction semantics;
- action alphabet and permitted/observed action contexts;
- transition family and parameters: table, affine map, polynomial, finite automaton, guarded relation, graph, or operator composition;
- observer outputs and projected state equivalence contract;
- known, ambiguous, or unobserved transition cells;
- validation coverage, witnesses, failures, costs, and confidence classification;
- compact realization instructions and a verifier identity, where available.

Generated models must not automatically become trusted executable code or grant authority. Require an independent verification route and explicit promotion policy. Keep the raw source snapshot addressable even when the Ground is an abstraction.

## C. Bottom-up physical host laboratory

### Stage 0 — Establish a reproducible local environment

**Initial assumption:** Windows is the first target; the same capture manifest format must also support Linux adapters. Inventory the machine rather than assuming its exact hardware, drivers, available privilege level, or GPU. Use ordinary-user access first.

1. Create a separate \`vl64-lab\` workspace with an isolated Python 3.11+ environment. Keep experimental dependencies separate from the standard-library-only \`vld\` package.
2. Clone VL-64 at a pinned commit and run \`python -m pip install -e ".[dev]"\`, \`python -m vld verify\`, and \`python -m pytest -q\`. Record the commit, operating system, Python version, CPU model, available memory, GPU/driver, display configuration, and clock source.
3. Create directories: \`sessions/\`, \`blobs/\`, \`models/\`, \`reports/\`, \`fixtures/\`, and \`adapters/\`. Each session has a unique ID, manifest, environment fingerprint, operator provenance, and report.
4. Use test programs, synthetic documents, and a dedicated local HTTP server as capture targets. Enable an explicit allowlist of processes, windows, interfaces, and paths. Never start with unrestricted system-wide screen or network recording.
5. Begin with a closed-loop program whose complete state is independently observable and whose operations are controlled. This supplies a known truth reference before moving to real desktop applications.

**Gate G0:** Full existing VL-64 tests pass on the actual computer, an empty capture session round-trips through the schema, and raw files can be reopened and integrity-verified in a new process. Publish a machine inventory and performance baseline without secrets or personal screen/network contents.

### Stage 1 — Software states, actions, and process lifecycle

Start with a tiny Python process with a 16-bit counter, mode flag, queue length, and pseudo-random/noise field. Add actions \`increment\`, \`reset\`, \`enqueue\`, \`dequeue\`, and \`toggle\`. The process should expose both an exact state endpoint and a byte-serialized snapshot, with known semantics for independent evaluation.

- Collect \`before/action/after\` triples, plus snapshots at passive times with no action.
- Exercise constant, increment, modular, branch, and state-dependent transitions; vary action order to separate correlations.
- Trace creation/exit and scheduling-related events using local process instrumentation. On Windows, prefer ETW or Windows Performance Recorder for OS traces; start with an instrumented child process for ground-truth variables. On Linux, optionally \`perf\`/tracepoints/eBPF with least privilege.
- Reproduce the same behavior in two process lifetimes, with new IDs and separate clock scopes. Test restart, reset, and lost-observation cases.
- Emit both a full-byte baseline and typed field candidates from a declared schema, and require the lifting/reconstruction invariants to be checked for exact fields.

**Gate G1:** At least two separately launched processes produce independently verifiable state traces. An action-aware model predicts held-out transitions more accurately than the raw byte-offset baseline, without treating time or accidental noise as a causal control variable.

### Stage 2 — Hardware and device observations

Observe machine behavior at three different levels rather than equating them:

1. **Hardware inventory:** device/CPU/GPU characteristics, driver versions, memory topology, device identities and exposed capabilities.
2. **Measured counters:** CPU usage, per-process memory and I/O, storage reads/writes, GPU engine activity where exposed, temperatures and power if an available supported API supplies them.
3. **Demanded computation:** an instrumented workload with known input and result, e.g. matrix-vector products, compression, file hashing, and bounded GPU kernels. The workload establishes what the measured activity corresponds to.

On Windows, use system performance counters/PDH, ETW and user-mode instrumentation; use GPU vendor counters only when supported by the installed driver. Avoid guessing internal register semantics from utilization counters. On Linux use \`/proc\`, \`/sys\`, perf/eBPF and driver counters through separate adapters.

Capture metadata must label units and aggregation windows; a rate at one sampling period is not a discrete transition at another. Log scheduler and thermal changes so models do not mistake background load for program logic.

**Gate G2:** A workload's requested operation can be linked to typed input, typed output, timing, and hardware observations, with an independent exact or tolerance-based reference. Hardware counters remain observations rather than proofs of every internal instruction.

### Stage 3 — Screens, pixel buffers, and visual state

Use an application built for the lab: a counter, toggles, progress bar, simple drawing canvas, and stable UI text. The same app must publish its internal state over a test endpoint, creating a separate ground-truth oracle.

On Windows, evaluate [Windows.Graphics.Capture](https://learn.microsoft.com/en-us/windows/uwp/audio-video-camera/screen-capture) for a permitted window, and [DXGI Desktop Duplication](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api) for GPU-backed desktop frames. Capture source bounds, display scale, frame size, pixel format, stride, color space, and frame timestamp. On Linux, use the desktop's supported capture/portal API.

Start with byte-exact pixel comparisons and region-of-interest/changed-tile hashes. Then add:
- stable element coordinates and change masks;
- connected components, template matching, shape/color features, and text extraction when necessary;
- UI Automation/accessibility tree as a separate structured observation source rather than inferred truth from pixels;
- frame-to-action alignment and explicit occlusion/resolution/multi-monitor cases;
- semantic candidate variables: button state, progress value, cursor target, object count and app mode.

Correlate image-derived variables with the controlled app's exact state. Use held-out themes, window locations, font scales, and occlusions to detect overfitting. Screenshots are useful observations but alone do not prove unseen hidden application state.

**Gate G3:** The Ground predicts the visible effect of at least three distinct authorized UI actions; captures preserve coordinate/scale provenance; a second theme or window geometry tests transfer. The prediction should be reproducible from the recorded video/frames and action receipts.

### Stage 4 — Communication streams as state-changing systems

Begin with loopback only: a dedicated local HTTP server and a controlled client. Add TCP framing/sequence events, request/response metadata, DNS lookup responses as observable records, and Git protocol packet-line parsing as structured data.

- Record request, admission by the receiving program, processing, response, and interpretation separately.
- Use an application-level sequence ID, trace ID, response status, timing and payload digest. Model retries, out-of-order arrivals, idempotence and deliberate timeouts.
- On Windows, evaluate [Pktmon](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/pktmon) for controlled packet captures; retain raw packet bytes only when necessary and permissioned. On Linux, use a filtered, local-only packet capture.
- Separate the network path's actual transformation from computation that took place at endpoints. Distinguish wire-frame evidence from an application-level assertion about what happened.
- Add an isolated second host only after loopback reconciliation and clocks/permissions are explicit. Never assume timestamps from separate machines share a monotone clock.

**Gate G4:** The same requested state transformation is traceable through raw application observations, captured communication, and an independent result checker. Retries and negative controls cannot be misreported as successful computation.

## D. Discovering relationships beyond byte offsets

### D1. Candidate representation ladder

Apply progressively richer, explicitly typed description families to the same grounded observation corpus.

| Level | Candidate family | Question to answer |
| --- | --- | --- |
| L0 | Exact byte windows and digests | What bytes were observed? |
| L1 | Endian integers, strings, structured records, image tiles, packet fields | Which reversible interpretations fit? |
| L2 | Temporal deltas, periodic fields, bitsets, counters, checksums, array/field dependencies | What changes with what? |
| L3 | Action-conditioned finite transition systems and guarded rules | What does each permitted intervention change? |
| L4 | Affine/linear maps, modular laws, sparse polynomials, recurrences, operator factorizations | Which compact law predicts many transitions? |
| L5 | Observer-equivalence quotients, composed Grounds, dependency DAGs | Which internal differences can be discarded without changing a relevant future observation? |
| L6 | Qualified navigators, indexed cycles, monoid jumps, checked realizers | Can the desired future state be produced with less measured work? |

At each level, preserve an explicit observation projection \`O(state)\`. A transition model that predicts a private compressed label but cannot reconstruct or validate the required observable is not sufficient.

### D2. Search and hypothesis generation

Candidate generators should be bounded, deterministic, and budgeted. Begin with known formats and alignment; expand to sliding windows, endian choices, differences, bit masks, periodicity, small tuples, and correlated fields. Then test finite-state machines, per-action transitions, affine recurrences, sparse polynomial lifts, and cross-field dependencies.

Use change-point tests and causal interventions to distinguish background correlations from action effects. A candidate relation should state where its variables came from, how it was inferred, the sample count, the actions under which it holds, and what remains unobserved.

Propose a family before measuring it on final test data. Limit candidates, description size, degree, operator depth, search CPU time, and capture bytes. Log how many candidates were considered and omitted.

### D3. Active observation and distinguishing experiments

When two candidate descriptions predict the same current observation but disagree after a particular action sequence, choose that sequence as the next *authorized* test. Prioritize by information gain per unit cost and risk, not by curiosity alone.

Reuse the finite-state distinguishing and partial-attention algorithms already present in the research repositories. When a system exposes only passive observation, label distinctions unresolved; do not invent interventions.

Example: a progress bar might be a timestamp animation or a queue-driven indicator. Waiting, issuing \`enqueue\`, and stopping the worker provide different predictions. The lab should choose a permitted, minimal discriminating experiment and record its result.

### D4. Predictive selection rather than maximum apparent state count

Compare development/train, later validation, and untouched test sessions in chronological order. Reserve complete new process lifetimes and UI geometries as transfer tests. Avoid leakage from repeated near-identical screen frames or machine-specific counters.

Report separately:
- coverage of supported states/actions, prediction accuracy on covered cases, and **predictive yield** over all requested cases;
- observed state diversity, reachable-state support sampled under interventions, and distinct correctly predicted states;
- transition-value match and change-detection accuracy;
- conditional entropy, predictive information, branching uncertainty, and residual/unmodeled fields;
- description bytes, witness size, preparation work, observation cost, navigation cost, recovery overhead, and verification cost;
- results of exact whole-domain checks where a finite state space is genuinely enumerable.

Use Pareto comparison; retain candidates that trade reachability, predictive yield, description size, and measured cost differently. Do not collapse them into a single secret score. \`ObservedGround.evaluate\` remains a useful initial heuristic, not the ultimate model-selection authority.

## E. Navigation, evidence, and runtime promotion

### E1. Classification of claims

- **Observed:** directly captured bytes/events with acquisition provenance.
- **Interpreted:** typed variables under a declared decoder, units, and round-trip/verification rule.
- **Inferred:** a law fitted from finite data with explicit support and withheld tests.
- **Certified within a finite domain:** fully checked transition/equivalence claim over that declared finite domain or validated mathematical identity under stated assumptions.
- **Executable/admitted:** verified description plus explicit local policy to use an operator on a qualified host.

Never promote empirical coverage into whole-domain proof. A model-generated future state does not become independent observation merely because the model also generated its receipt.

### E2. Exported Ground package

Proposed package layout:

~~~text
ground-package/
  ground.json            # versioned typed variables, actions, laws, scope
  observation.jsonl      # sampled evidence references and action ids
  evidence.json          # acquisition/certificate/validation summary
  operators.json         # finite or algebraic operation descriptions
  benchmark.json         # cost model, reference comparisons and environment
  blobs/                 # only necessary raw data, content-addressed
~~~

Use canonical hashing, typed refusal outcomes, independent revalidation of inputs and claims, and optional signatures when actor identity matters. The compact VL-64 reader should need only \`ground.json\`, necessary small witness/parameter data and the small verifier; the host lab retains large artifacts.

### E3. Admission and realization rules

1. Verify schema, provenance, source digests, action context and declared Ground identity.
2. Verify a candidate law on specified finite witnesses and separate test data; use full-domain proofs only where available.
3. Determine whether the intention needs exact semantics, approximate semantics with a tolerance, or a refusal on unknown states.
4. Choose among admitted operators by componentwise cost and qualification, not a marketing speedup estimate.
5. Execute navigation and independently check the observable terminal result or an accepted proof/certificate.
6. Record a receipt with input, Ground, operator, host, result, verification method, and actual measured work.
7. Require new source observations before treating prediction on a previously unobserved physical state as established.

**Security checkpoint:** An earlier empirical audit of VL-64 identified acceptance of duplicate quorum signatures and unsupported certificate kinds. Resolve and regression-test those findings before the experimental lab relies on quorum/standalone bundle verification for security-critical authority. Raw hashes bind content, not the honesty of the observer.

## F. Experiment campaigns and objective gates

### Experiment X1 — Controlled finite software machine (first priority)

- **System:** instrumented, independently specified counter/queue/mode simulator; exact binary snapshots and action receipts.
- **Baseline:** full bytes + \`ObservedGround\` offset views + direct step-by-step simulation.
- **Models:** typed fields, action-conditioned finite transition table, affine modulo components, quotient classes.
- **Interventions:** reset, branch, queue operations, unseen action orders, restart, deliberate dropped observations.
- **Evaluation:** held-out complete episodes and exhaustive small-state checks when tractable.
- **Pass:** exported Ground agrees on every tested exact transition, refuses unknown action/state cells, and shows either higher predictive yield or lower measured work than baseline. No claim of universal induction from random finite sampling.

### Experiment X2 — Window and UI machine

- **System:** small locally controlled GUI with published internal state, switch, counter, progress, render loop.
- **Baseline:** raw frames and changed pixel rectangles, no typed GUI semantics.
- **Models:** pixel regions, UI Automation identity, action-conditioned visible state changes, display-independent abstraction.
- **Interventions:** click, keypress, resize, theme change, occlusion.
- **Pass:** exact or declared-tolerance visible-state predictions on separate sessions, including at least one changed display geometry; evidence connects frames to user-authorized actions.

### Experiment X3 — Controlled network process

- **System:** local JSON/HTTP service with explicit state transition contract; optional Git packet-line fixture.
- **Baseline:** raw request/response blobs and endpoint-side direct operation.
- **Models:** typed messages, request-conditioned server state, causal event sequence, retry/idempotence relation.
- **Interventions:** duplicate request, malformed request, delayed response, timeout, new client connection.
- **Pass:** predicted application state and communication receipts agree with independent server logs; timeouts remain \`UNKNOWN\` rather than negative data.

### Experiment X4 — CPU/GPU numerical host

- **System:** repeatable matrix/vector operations and small structured tensor fixtures, with independent reference results.
- **Baseline:** dense arithmetic and existing VL-64 structural operator implementation.
- **Models:** rank/sparsity/periodicity, typed tensor dimensions and codecs, operator graph and input/output dependencies.
- **Interventions:** vary data size, rank, precision, shape, process placement, and warmed vs cold execution.
- **Pass:** declared error tolerance and reproducible cost/byte advantages at specified scales; total cost includes discovery, preparation, and verification when comparison requires them.

### Experiment X5 — Cross-substrate description transport

- **System:** same finite intention and observable realized in an in-process interpreter, separate child process, and one permissioned communication-backed service.
- **Baseline:** direct realization per host.
- **Models:** same exported Ground and observer contract, distinct host-specific execution adapters.
- **Pass:** equivalent independent outputs and grounded receipts across host topology changes; unsupported realizers refuse rather than falsely claiming equivalence.

## G. Benchmark and reporting rules

Track the following per experiment without combining them into an unqualified "speedup":

| Metric | Method |
| --- | --- |
| Acquisition cost | Elapsed monotonic duration, CPU usage, bytes captured, lost events |
| Translation cost | Parse/inference time, peak memory, candidate count, source bytes |
| Prediction | Test coverage, accuracy on covered cases, predictive yield, exact transition match |
| Information | Observed support, conditional entropy, distinguishability, residual fields |
| Navigation | Preparation steps, lookup/compose steps, actual elapsed time, peak memory |
| End-to-end work | Capture + translation + planning + verification + realization, amortized over reuse count |
| Reuse | Performance on new actions, process lifetimes, display arrangements, host configurations |
| Correctness | Independent-reference agreement, whole-domain proof scope, false admissions, refused unknowns |

Record p50/p95 and dispersion over repeated runs; warm-up and runner load must be declared. Compare like for like, including the expense of learning the Ground. A trillion-step jump is meaningful only when its transition law is valid for the intended domain; end-to-end advantage must account for acquiring that law.

Negative controls must include reordered timestamps, mislabeled actions, random/noise bytes, mismatched data formats, adversarial record edits, missing transitions, dropped frames, stale calibration, and a competing candidate that looks good on training data but fails held-out cases.

## H. Deliverables and execution order

| Phase | Product | Move forward when |
| --- | --- | --- |
| P0 | Lab inventory, capture schema, isolated fixtures, baseline report | G0 proves reproducible acquisition |
| P1 | Process and action adapter; exact state Ground | G1 demonstrates held-out action-conditioned improvement |
| P2 | Typed decoder registry and candidate search | Reversible fields are checked; non-reversible fields are explicitly labeled |
| P3 | Finite-law and probe generator | X1 verified against independent transitions, including refusals |
| P4 | Screen capture and semantic visual adapter | X2 passes with new geometry/theme |
| P5 | Network/communication adapter | X3 preserves separate transport and application evidence |
| P6 | Hardware and tensor observation adapter | X4 produces validated numerical/cost comparisons |
| P7 | Compact Ground export and navigation verifier | X5 realizes the same intention across distinct hosts |
| P8 | Adaptive model-selection loop | A later, untouched session demonstrates improvement over prior models |

**First implementation milestone:** P0–P3 on a single controlled process. This is the quickest way to find whether the translation mechanism can move beyond raw byte offsets while retaining independent ground truth.

**Immediate next build:** add the *external* capture-envelope library, controlled finite-state test process, and replayable corpus generator; benchmark \`ObservedGround\` against typed/action-conditioned descriptions on that corpus. Do not touch the 64 KiB core until the external experiment demonstrates an operation worth compressing into it.

## I. Practical first session on a Windows host

Run from PowerShell (commands describe the current repository installation; the proposed \`vl64-lab\` capture CLI does not exist yet):

~~~powershell
git clone https://github.com/Kier73/VL-64.git
cd VL-64
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m vld verify
.\.venv\Scripts\python.exe -m pytest -q
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfLogicalProcessors
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
~~~

Capture inventory as local research metadata, redact device serials, user paths, screen content and network payloads before sharing any report. Start with the controlled process simulator, then attach OS observation tools to that process rather than collecting unrelated user activity. Some ETW providers, packet tools and GPU counters require additional permissions or installation; check on the actual host.

### Relevant platform documentation

- Windows screen capture: https://learn.microsoft.com/en-us/windows/uwp/audio-video-camera/screen-capture
- DXGI Desktop Duplication: https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api
- Event Tracing for Windows: https://learn.microsoft.com/en-us/windows/win32/etw/about-event-tracing
- Windows performance counters: https://learn.microsoft.com/en-us/windows/win32/perfctrs/about-performance-counters
- Packet Monitor: https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/pktmon

## J. The research question to preserve

**Given a finite set of observations and permitted interventions on an actual system, which compact, independently checkable mathematical representation enables the largest useful family of correct predictions and realizations at the lowest measured total cost?**

Repeated observation, new actions, new substrates, and changed observer contexts should improve the candidate Grounds and their tests. Promotion is earned by evidence, not by an attractive abstraction. The intended outcome is a virtual description that becomes an increasingly effective way to use the observed system.
