# VL-64: 64 KiB Virtual Layer Runtime

A **64 KiB Virtual Layer Utility**.

VL-64 translates captured bytes into candidate virtual state representations, measures which descriptions predict observed transitions, and navigates qualified states through composition and cycle detection. It also provides mathematical, structural, and verification utilities for performing work on those representations. Its source package is maintained within a 65,536-byte budget. It runs on Python 3.11+ using the standard library.

Use the `vld` command for built-in checks and modular affine navigation, or import `vld` to compose the available operations in your own code.

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

## Documentation

- [Usage guide](docs/USAGE_GUIDE.md) — commands, API examples, and extension points.
- [Research goals](docs/RESEARCH_GOALS.md) — the functional experiments and verification criteria guiding further development.
- [Observer intention and model Grounds](docs/OBSERVER_INTENTION_AND_MODEL_GROUNDS.md) — the typed input contract, evidence, and remote-operation research path.
- [Generic byte fabric](experiments/byte_fabric/README.md) — architecture-independent byte observation, decoder plug-ins, and qualified execution.
- [Compressive Virtual Grounds](docs/COMPRESSIVE_VIRTUAL_GROUNDS.md) — discover a predictive byte interpreter, compress observed state histories, and compose distant-state navigation.
