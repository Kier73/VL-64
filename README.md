# VL-64: 64 KiB Virtual Layer Runtime

A **64 KiB Virtual Layer Utility**.

VL-64 translates captured bytes into candidate virtual state representations, measures which descriptions predict observed transitions, and navigates qualified states through composition and cycle detection. It also provides mathematical, structural, and verification utilities for performing work on those representations. Its source package is maintained within a 65,536-byte budget. It runs on Python 3.11+ using the standard library.

Use the `vld` command for built-in checks and modular affine navigation, or import `vld` to compose the available operations in your own code.

The standalone model seed can start at a Hugging Face search result, model card or directory, bind the displayed commit and evaluate supported Llama decoders from token IDs. The page-enabled archive occupies 26,723 bytes. [Usage and measured costs](docs/STANDALONE_MODEL_SEED.md)

[How a worker addresses a hosted model](docs/HOSTED_MODEL_ADDRESSES.md) follows a real Qwen embedding from its tensor name to a reproducible 10,240-byte read.

## Install

```bash
git clone https://github.com/Kier73/VL-64.git
cd VL-64
python -m pip install -e .
```

## Command-line use

```bash
vld verify
vld info
vld navigate --a 5 --b 1 --x0 3 --steps 1000 --modulus 256
```

- `verify` prints a JSON report of five built-in checks: Ed25519, affine navigation, structured linear operation, quorum certificate verification, and compiler realization.
- `info` prints the installed runtime version information.
- `navigate` evaluates `x[t+1] = (a*x[t] + b) % modulus` after the requested number of steps, using composition rather than a step-by-step loop.

You can also call the CLI as `python -m vld` after installing the package.

## Use from Python

### Translate observed bytes into a Ground

`ObservedGround` lifts captured snapshots into named byte views, measures their observed transition patterns, and navigates known deterministic cycles.

```python
from vld import ObservedGround

ground = ObservedGround({"whole": None, "counter": (0,)})
ground.fit(
    (bytes((i, 10)), bytes(((i + 1) % 4, 20)))
    for i in range(4)
)

observation = ground.lift(bytes((2, 99)))
prediction = ground.predict(bytes((2, 99)), "counter")
result = ground.navigate(bytes((2, 99)), steps=1_000_000, view="counter")

assert observation["states"]["counter"] == "02"
assert prediction["status"] == "SINGLE_OBSERVED_SUCCESSOR"
assert result["terminal_state"] == "02"
print(ground.profile("counter"))
```

To discover candidate byte fields automatically, call `model, ranking = ObservedGround.discover(training_pairs, heldout_pairs)`. The search tests bounded single-byte and multi-byte views, then ranks them using held-out predictions.

A view may describe the entire snapshot or select byte offsets corresponding to a variable, register, field, or observable. `evaluate` compares candidate views against held-out transitions and ranks them by observed predictive score. Information profiles report state diversity, branching uncertainty, and changes in empirical state entropy. Navigation returns the projected virtual state and the digest of its observed Ground.

### Describe an observer intention

`ObserverIntention` turns a bounded, structured request into a content-addressed contract. The core stores the observer's requested operation, target, output format, and options; external adapters decide how to realize it.

~~~python
from vld import ObserverIntention

request = ObserverIntention(
    "bytes.observe",
    "workspace:experiments/byte_fabric/fixtures/state.json",
    output="json",
    options={"include_data": True},
)
print(request.digest)
~~~

The [generic byte-fabric lab](experiments/byte_fabric/README.md) takes this request to a qualified file, byte stream or HTTPS source, builds an exact byte Ground, and optionally attaches typed views. `bytes.execute` can request registered, bounded operations such as SHA-256, byte slicing or JSON-pointer extraction, with an execution receipt. Unknown formats remain valid byte Grounds, and host-installed plug-ins can recognize additional architectures without changing the core.

To describe **the result rather than the operator**, use `bytes.realize` with a typed goal, such as `value.json` or `digest.sha256`. The host selects a qualified relation and produces a replayable execution receipt. See [Goal-directed byte navigation](docs/GOAL_DIRECTED_BYTE_NAVIGATION.md).

For example, run `python -m experiments.observer_intent.arbitrate --intent experiments/byte_fabric/intents/host_capabilities.json --out capabilities.json` to ask an observer-host what sources, decoders and operators it supports. Specialized providers, including the earlier [model-inspection lab](experiments/observer_intent/README.md), use the same intention contract.

### Navigate a modular state

```python
from vld import affine_navigate, certify_modular_progression

state = affine_navigate(a=5, b=1, x0=3, n=1_000_000, m=256)
print(state)

is_identified, parameters = certify_modular_progression([240, 250, 4, 14])
assert is_identified and parameters == (10, 256)
```

`affine_navigate` composes repeated affine transitions in logarithmic time and compares two derivations of the result. `certify_modular_progression` checks an observed modular step sequence.

### Compile and realize an affine intention

```python
from vld import IntentionCompiler, IntentionSpec, IntentExpr

compiler = IntentionCompiler()
spec = IntentionSpec(
    intention_id="example",
    target_expr=IntentExpr.iterate_affine("x", a=5, b=1, steps=1000, modulus=256),
    contract_kind="int_vector",
    lineage_id="example-session",
)
plan = compiler.compile(spec)
receipt = compiler.realize(plan, {"x": [3]})

print(plan["selected_plan"])
print(receipt["admitted_observation"])
```

The compiler selects between direct stepping and an affine monoid jump using its cost model. `realize` checks the resulting value against affine navigation before returning an admission receipt. Optional `NavigationCostVector` ceilings can constrain plan selection.

### Work with bytes and forked state

```python
from vld import LENS_UTF8, PageSpace

data = b"Hello, VL-64"
assert LENS_UTF8.verify_roundtrip(data)
assert LENS_UTF8.lower(LENS_UTF8.lift(data)) == data

parent = PageSpace()
parent.write_page(0, 0, b"initial")
child = parent.fork()
child.write_page(0, 0, b"updated")

assert parent.read_page(0).startswith(b"initial")
assert child.read_page(0).startswith(b"updated")
print(child.commit_manifest()["digest"])
```

## Available utilities

| Area | Python interfaces | Function |
| --- | --- | --- |
| Translation | `ObservedGround` | Lift byte snapshots, compare predictive views, profile uncertainty, and navigate observed transitions |
| Observer input | `ObserverIntention` | Describe a target operation and output format with a stable request digest |
| Representation | `ByteLens`, `LENS_RAW`, `LENS_UTF8`, `LENS_U32LE` | Lift and lower byte representations; check round trips |
| State and time | `PageSpace`, `HashChainedJournal`, `ScopedVectorClock` | Copy-on-write pages, event digests, and scoped event ordering |
| Navigation | `monoid_power`, `affine_navigate`, `certify_modular_progression`, `myhill_nerode_quotient` | Compose transitions and analyze finite-state behavior |
| Exact reasoning | `ExactFieldMatrix`, `solve_sheaf_cohomology`, `syndrome_decode_sparse_fault`, Kleene K3 operators | Rational/finite-field reduction, consistency tests, and three-valued guards |
| Execution control | `IntentionCompiler`, `NavigationCostVector`, `EffectLedger`, `GroundRegistry` | Select affine plans, deduplicate effects within a ledger instance, and register grounds/lenses/rules |
| Certificates | `ed25519_sign`, `ed25519_verify`, `QuorumAuthority`, `verify_bundle_standalone` | Sign and check messages, local quorum certificates, and supported audit bundles |
| Numerical operations | `StructuralLinearOperator`, `LoRAMorphism`, FP8/FP4 decoding, `rms_norm`, `FlashStreamState` | Evaluate structured matrices, low-rank deltas, quantized values, normalization, and attention accumulators |
| Adapters and inspection | `GitWireV2Parser`, `SafetensorsGlance`, `HeadlessOS`, `StreamPathMachine` | Parse packet lines, inspect supplied tensor headers, manage simple process state, and transform byte streams |

The full examples, including certificates, structured operators, model metadata, stream operations, and extension points, are in the [usage guide](docs/USAGE_GUIDE.md).

## Run tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

The tests exercise the Python API and check the source package against the 64 KiB budget.

## Navigate a learned, compressed Ground

The [Compressive Virtual Ground experiment](docs/COMPRESSIVE_VIRTUAL_GROUNDS.md) discovers a scalar interpreter from changing byte snapshots, validates a small transition law against held-out observations, and retains one byte template plus its navigation rule instead of every frame.

The resulting Ground can answer a distant observer intention through **interpret → navigate → reconstruct → verify**, without replaying intermediate frames. For example:

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/ground_trillion.json \
  --out result.json
~~~

This sample navigates a saved fixed-layout, modular-affine Ground across one trillion *modeled* transitions. The measured description reduced 81,920 bytes of observed frames to 1,947 bytes. Its future behavior is a validated hypothesis rather than a guarantee about every possible machine.

## Discover a non-affine or observable Ground

VL-64 can now compare **multiple state-model families** against held-out byte observations. A new finite-state learner recognizes a witnessed closed non-affine cycle, retaining one byte template and a small state table instead of repeated snapshots. A second mechanism merges states only when their **entire modeled observable future** remains equivalent.

Use the existing observer interface with either saved Ground:

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/nonaffine_trillion.json \
  --out result.json
```

For an observation-only result, use `observable_quotient_trillion.json`. That smaller Ground can answer the observer's future-state question but **cannot reconstruct discarded internal bytes**. See [Non-Affine and Observable Grounds](docs/NONAFFINE_AND_OBSERVABLE_GROUNDS.md) for exact validation and limits.

## Ask the Virtual Layer what to observe next

VL-64's [active action-conditioned Ground experiment](docs/ACTIVE_ACTION_GROUNDS.md) can compare several declared finite-state descriptions, select a short permitted action sequence that best distinguishes them, and use the **actual observed output bytes** to eliminate inconsistent candidates. A contradiction is retained rather than misreported as a successful Ground.

```bash
python -m experiments.byte_fabric.active_lab --out artifacts
python -m experiments.observer_intent.arbitrate \
  --intent artifacts/active-next-probe-intent.json \
  --out artifacts/recommended-probe.json
```

The observer-facing `ground.active.plan` request recommends an informative experiment without executing it. Actual action execution requires a separately installed qualified host. The `ground.action.navigate` request can later navigate a saved model through a trillion **modeled** repeated transitions without sending those actions to an external service.

## Learn candidate Grounds from action traces

VL-64 can now generate **action-conditioned Ground hypotheses from observed byte traces**, rather than requiring a user or host to supply every candidate transition table. It checks observed input/output consistency, searches bounded state mergings, proposes distinguishing probes, and resynthesizes after new evidence arrives.

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/trace_derived_next_probe.json \
  --out next-experiment.json

python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/trace_derived_trillion.json \
  --out modeled-result.json
```

The first command **recommends** an experiment from two saved input/output traces; it does not execute host actions. The second navigates the learned finite-state model, rather than physically running a trillion operations. The [Trace-Derived Action Grounds](docs/TRACE_DERIVED_ACTION_GROUNDS.md) document explains the evidence, 62 independent held-out checks, and limits of model inference.

## Procedurally generate a monolithic VL-64 seed

The [Monolithic Seed and Self-Grounds](docs/MONOLITHIC_SEED_AND_SELF_GROUNDS.md) experiment derives a self-contained Python archive from the actual seven-module compact runtime. It carries **the exact source bytes, a verifying bootstrap, a source manifest, and a source-derived Ground of seven module Grounds**. The seed can be transferred to another Python-capable host and executed without installing the VL-64 package.

```bash
python -m experiments.seed_fabric.build build --out artifacts/VL64.pyz
python -I artifacts/VL64.pyz seed-verify
python -I artifacts/VL64.pyz seed-ground
python -I artifacts/VL64.pyz seed-relations
python -I artifacts/VL64.pyz verify
```

Both the expanded source and the archive itself must fit within **65,536 bytes**. The builder and tests live outside the runtime; the exact software Ground is generated from AST and file SHA-256 evidence, not a hand-maintained map. It describes verifiable software structure, not a universal interpreter or an autonomous Internet processor.

## Observe only the bytes that distinguish a Ground

VL-64's [Multi-Field and Partial-Observation Grounds](docs/MULTIFIELD_PARTIAL_GROUNDS.md) experiment discovers **multiple disjoint changing byte spans**, retains the witnessed relationships among their combined states, and navigates qualified action transitions. A partial observer can provide **just one known byte** and ask VL-64 which additional field is most informative to inspect next.

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_recommend_field.json \
  --out suggested-bytes.json

python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_trillion.json \
  --out modeled-result.json
```

This uses a recorded multi-field Ground; it performs no external actions. Missing transitions stay unresolved. The controlled 512-byte frame experiment compressed 16,384 bytes of observed transition frames into a 4,322-byte Ground and checked 62 new mixed action compositions. This is a *bounded fixed-layout prototype*, not a general arbitrary machine decoder.

## Observe partial state across several byte fields

The [Multi-Field and Partial-Observation Grounds](docs/MULTIFIELD_PARTIAL_GROUNDS.md) experiment discovers disjoint changing byte regions from actual frames, retains a **sparse, witnessed joint state/action graph**, and identifies the best missing byte interval for an external observer to inspect next.

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_recommend_field.json \
  --out suggested-observation.json
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_trillion.json \
  --out modeled-state.json
```

The observer may supply **one known byte**, not a whole frame. A requested missing transition remains unresolved; long-horizon navigation is confined to already witnessed edges. The experiments and adapter stay outside the 64 KiB runtime.

## Run language models and pretrained tensor operations through VL-64

The [Model Fabric](experiments/model_fabric/README.md) now executes two actual local model families **using the existing 64 KiB runtime operators**: a byte-level statistical language model trained on repository text, and a small weight-backed causal-attention decoder. Both have independent numerical/predictive tests and typed observer-intention requests.

It can also inspect a real pinned Hugging Face model through bounded Safetensors header requests, identify its tensor structure **without downloading weights**, and then **execute a genuine pretrained model sub-operation** by fetching only the required 5,760 BF16 weight bytes.

A [38,360-byte model seed](docs/MODEL_EXECUTION_GROUNDS.md) procedurally combines the original VL-64 runtime and the verified pretrained tensor slice into a single executable Python archive:

```bash
python -m experiments.model_fabric.model_seed \
  --out artifacts/VL64-ModelSeed.pyz
python -I artifacts/VL64-ModelSeed.pyz model-verify
python -I artifacts/VL64-ModelSeed.pyz model-run
```

The portable 38 KiB seed executes first-layer RMS normalization and **four Q-projection rows**. A separate [typed pretrained layer adapter](docs/PRETRAINED_LAYER0_GROUND.md) now executes a **complete first-layer decoder pass** over genuine qualified BF16/F32 weight ranges: token embeddings, RoPE/GQA, attention, SwiGLU and both residuals, with an independent scalar reference and cached weights across positions. The optional pinned-model verification acquires approximately 7.08 MB of first-layer BF16 parameters through bounded HTTPS ranges and refuses to claim success without exact bytes. This still does **not** establish full 30-layer language generation. The 65,093-byte compact source is unchanged.

See [Language Models, Pretrained Tensor Navigation and the Model Seed](docs/MODEL_EXECUTION_GROUNDS.md) for results, evidence and reproducible commands.

## Documentation

- [Usage guide](docs/USAGE_GUIDE.md) — commands, API examples, and extension points.
- [Research goals](docs/RESEARCH_GOALS.md) — the functional experiments and verification criteria guiding further development.
- [Observer intention and model Grounds](docs/OBSERVER_INTENTION_AND_MODEL_GROUNDS.md) — the typed input contract, evidence, and remote-operation research path.
- [Generic byte fabric](experiments/byte_fabric/README.md) — architecture-independent byte observation, decoder plug-ins, and qualified execution.
- [Compressive Virtual Grounds](docs/COMPRESSIVE_VIRTUAL_GROUNDS.md) — discover a predictive byte interpreter, compress observed state histories, and compose distant-state navigation.
- [Non-Affine and Observable Grounds](docs/NONAFFINE_AND_OBSERVABLE_GROUNDS.md) — discover finite-state transitions and compress according to future-stable observer equivalence.
- [Active Action-Conditioned Grounds](docs/ACTIVE_ACTION_GROUNDS.md) — select informative host-qualified observations, refine finite hypotheses, and navigate retained models.
- [Trace-Derived Action Grounds](docs/TRACE_DERIVED_ACTION_GROUNDS.md) — synthesize competing finite action models from byte traces, actively distinguish them, and test modeled navigation.
- [Monolithic Seed and Self-Grounds](docs/MONOLITHIC_SEED_AND_SELF_GROUNDS.md) — derive a single executable seed and verifiable Ground of the runtime's own modules and source relations.
- [Multi-Field and Partial-Observation Grounds](docs/MULTIFIELD_PARTIAL_GROUNDS.md) — discover interacting byte regions, represent missing transitions, and request only informative bytes.

### Recoverable pretrained model orbit

The [recoverable pretrained model orbit](docs/PRETRAINED_ORBIT_RECOVERY.md) uses SHA-256-addressed, 64-KiB-or-smaller KV pages and externally pinned checkpoints to resume the verified two-layer pretrained transformer across a new observer-host assignment without replaying accepted tokens. The independent fixture suite tests a fresh Python process, stale/altered evidence rejection, BF16/F32 agreement and rollback. This is local compute with qualified range sources, not an attested remote processor or complete model inference.
