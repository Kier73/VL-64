# VL-64 Usage Guide

This guide shows how to use the packaged commands and Python interfaces. All examples use the repository's `vld` package.

## Install and verify

Python 3.11 or newer is required.

```bash
git clone https://github.com/Kier73/VL-64.git
cd VL-64
python -m pip install -e .
vld verify
```

The `verify` command prints JSON with `status`, `checks`, and `total_assertions`. Use `vld info` for the version string and `python -m vld verify` when you prefer module execution.

Run the broader test suite with:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Translate byte observations into a virtual Ground

`ObservedGround` treats each snapshot as bytes and allows multiple named views over them. A view with `None` preserves every byte. A tuple of offsets selects an explicit subset, so the source of each virtual variable remains inspectable.

The observer or adapter supplies snapshots from a machine, binary record, process, file, or another source. The core accepts bytes; collecting screenshots or network content is an adapter task.

```python
from vld import ObservedGround, GroundRegistry

ground = ObservedGround({
    "full_snapshot": None,
    "counter_register": (0,),
    "second_field": (1,),
})

# Observed training transitions: the first byte advances mod 4.
training = [
    (bytes((i, 16 * t + i)), bytes(((i + 1) % 4, 80 + 16 * t + i)))
    for t in range(4) for i in range(4)
]
ground.fit(training)

print(ground.lift(bytes((3, 250))))
print(ground.predict(bytes((3, 250)), "counter_register"))
print(ground.profile("counter_register"))
GroundRegistry.default().register_ground("observed-counter", ground)
```

The `lift` result includes a digest of the original bytes and each view's projected state. `predict` reports successor counts and one of `UNOBSERVED`, `SINGLE_OBSERVED_SUCCESSOR`, or `MULTIPLE_OBSERVED_SUCCESSORS`. `profile` reports empirical source and target entropy, conditional entropy, and target/source state diversity. These quantities characterize observations, not unobserved machine behavior.

### Compare alternate descriptions

Use different observations for training and validation. The score is the held-out target state entropy multiplied by the fraction of correct most-frequent-successor predictions. This rewards a view with distinct predictable outcomes rather than simply selecting a constant projection.

```python
heldout = [
    (bytes((i, 160 + i)), bytes(((i + 1) % 4, 192 + i)))
    for i in range(4)
]
ranking = ground.evaluate(heldout)
assert ranking[0]["view"] == "counter_register"
assert ranking[0]["accuracy"] == 1.0
print(ranking)
```

### Navigate a learned state transition

```python
result = ground.navigate(bytes((3, 250)), 10**12 + 5, "counter_register")
assert result["terminal_state"] == f"{(3 + 10**12 + 5) % 4:02x}"
assert result["evaluated_steps"] <= 4
assert result["evaluated_steps"] + result["skipped_steps"] == 10**12 + 5
print(result)
```

A deterministic observed cycle can be jumped with cycle indexing. The output is the projected state (hex), not a synthetic reconstruction of omitted bytes. If any required transition was not observed or has multiple observed successors, navigation raises `ValueError`. To examine a new file or captured machine snapshot, pass its bytes to `ground.lift(snapshot)`, then fit actual observed before/after pairs before predicting transitions.

---
 
## 1. Navigate an affine state

The CLI evaluates the state after `steps` applications of `x -> (a*x + b) % modulus`:

```bash
vld navigate --a 5 --b 1 --x0 3 --steps 1000000 --modulus 256
```

The Python interface also exposes the composition primitive:

```python
from vld import affine_navigate, certify_modular_progression, monoid_power

result = affine_navigate(a=5, b=1, x0=3, n=1_000_000, m=256)
assert 0 <= result < 256

# Identify an additive modular step from a trace with a witnessed wrap.
ok, step_and_modulus = certify_modular_progression([240, 250, 4, 14])
assert ok and step_and_modulus == (10, 256)

# Compose any associative operation with a supplied identity.
power = monoid_power(2, 10, lambda a, b: a * b, 1)
assert power == 1024
```

## 2. Select and realize a plan

`IntentionCompiler` currently compiles affine iteration expressions. The cost vector has four dimensions: estimated latency, operation count, bytes moved, and recovery overhead.

```python
from vld import (
    IntentionCompiler, IntentionSpec, IntentExpr, NavigationCostVector
)

spec = IntentionSpec(
    intention_id="orbit-request",
    target_expr=IntentExpr.iterate_affine("x", a=5, b=1, steps=1000, modulus=256),
    contract_kind="int_vector",
    lineage_id="run-1",
    cost_ceilings=NavigationCostVector(
        latency=1.0, compute_ops=100.0, byte_movement=64.0, recovery_overhead=0.0
    ),
)
compiler = IntentionCompiler()
plan = compiler.compile(spec)
assert plan["selected_plan"] == "monoid_jump"

receipt = compiler.realize(plan, env={"x": [3]})
assert receipt["status"] == "ADMITTED"
assert receipt["admitted_observation"] == [
    __import__("vld").affine_navigate(5, 1, 3, 1000, 256)
]
```

`compile` uses the package's current cost estimates to choose `direct_step` or `monoid_jump`. `realize` evaluates the chosen route and compares it with the affine reference operation.

## 3. Represent bytes and manage state

### Coordinate lenses

```python
from vld import LENS_RAW, LENS_UTF8, LENS_U32LE

raw = b"hello"
assert LENS_UTF8.lift(raw) == "hello"
assert LENS_UTF8.lower("hello") == raw
assert LENS_UTF8.verify_roundtrip(raw)

assert LENS_RAW.lift(bytes([1, 2, 3])) == [1, 2, 3]
assert LENS_U32LE.lift(bytes([1, 0, 0, 0])) == [1]
```

For `LENS_U32LE`, supply a byte length divisible by four when an exact round trip is required.

### Copy-on-write pages and manifests

```python
from vld import PageSpace

parent = PageSpace()
parent.write_page(0, 0, b"before")
child = parent.fork()
child.write_page(0, 0, b"after!")

assert parent.read_page(0).startswith(b"before")
assert child.read_page(0).startswith(b"after!")
manifest = child.commit_manifest()
print(manifest["pages"], manifest["digest"])
```

### Journal and vector clock

```python
from vld import HashChainedJournal, ScopedVectorClock, verify_bundle_standalone

journal = HashChainedJournal()
journal.record("OBSERVED", {"value": 7})
audit = verify_bundle_standalone({
    "kind": "journal_continuity",
    "genesis": "0" * 64,
    "entries": journal.entries,
})
assert audit["status"] == "VERIFIED"

a = ScopedVectorClock("scheduler-1")
b = ScopedVectorClock("scheduler-1")
a.tick("worker-1")
assert a.compare(b) == 1
```

## 4. Check observations and exact relations

### Three-valued interval guards

```python
from vld import (
    classify_interval_guard, k3_and,
    K3_TRUE, K3_FALSE, K3_INDETERMINATE
)

assert classify_interval_guard(0.1, 0.4, threshold=0.5) == K3_TRUE
assert classify_interval_guard(0.6, 0.8, threshold=0.5) == K3_FALSE
assert classify_interval_guard(0.4, 0.7, threshold=0.5) == K3_INDETERMINATE
assert k3_and(K3_TRUE, K3_INDETERMINATE) == K3_INDETERMINATE
```

### Sheaf consistency and sparse syndrome

```python
from fractions import Fraction
from vld import solve_sheaf_cohomology, syndrome_decode_sparse_fault

edges = [[-1, 1, 0], [0, -1, 1]]
ok, values = solve_sheaf_cohomology(edges, [Fraction(3), Fraction(4)])
assert ok
assert values[2] - values[0] == 7

triangle = [[-1, 1, 0], [0, -1, 1], [1, 0, -1]]
ok, witness = solve_sheaf_cohomology(triangle, [1, 1, 1])
assert not ok
print("Inconsistency witness:", witness)

# A single-column fault with syndrome (0, 3) modulo 7.
error = syndrome_decode_sparse_fault(
    [[1, 0, 1], [0, 1, 1]], [0, 3], p=7
)
assert error == [0, 3, 0]
```

## 5. Sign, inspect, and deduplicate operations

### Messages and quorum certificates

```python
from vld import (
    ed25519_keypair_from_seed, ed25519_sign, ed25519_verify, QuorumAuthority
)

# Fixed seeds are for a reproducible example, not operational key material.
seed = bytes([7]) * 32
_, public_key = ed25519_keypair_from_seed(seed)
signature = ed25519_sign(seed, b"example")
assert ed25519_verify(public_key, b"example", signature)

authority = QuorumAuthority([
    bytes([1]) * 32, bytes([2]) * 32, bytes([3]) * 32
])
certificate = authority.sign_proposal({"state": "ready", "epoch": 1})
assert authority.q_min == 2
assert authority.verify_quorum_certificate(certificate)
```

`QuorumAuthority` signs using local member keys supplied to the constructor. Certificate verification checks the signatures against those configured public keys.

### Effect ledger

```python
from vld import EffectLedger

ledger = EffectLedger()
events = []

def record_once():
    events.append("sent")
    return "sent"

assert ledger.execute_effect("intent-1", "event-1", record_once) == "sent"
assert ledger.execute_effect("intent-1", "event-1", record_once) == "sent"
assert events == ["sent"]
```

The ledger holds its journal and deduplication map in the Python process. Reuse the same instance to deduplicate the same `(intention_id, effect_key)` pair.

## 6. Evaluate structured and streaming operators

### Structured matrix-vector product and low-rank delta

```python
from vld import StructuralLinearOperator, LoRAMorphism

op = StructuralLinearOperator(rows=2, cols=3, alpha=1.0)
assert op.matvec([1.0, 2.0, 3.0]) == [6.0, 6.0]

adapter = LoRAMorphism(
    alpha=2.0, r=1,
    A=[[1.0], [2.0]], B=[[1.0], [1.0]]
)
assert adapter.contract([1.0, 1.0]) == [4.0, 8.0]
```

`StructuralLinearOperator` accepts affine row/column coefficients, an optional periodic orbit, low-rank `U`/`V` factors, and sparse `(row, column)` corrections.

### Combine attention accumulators

```python
from vld import FlashStreamState

left = FlashStreamState(m=1.0, l=1.0, O=[2.0])
right = FlashStreamState(m=2.0, l=1.0, O=[4.0])
combined = left.combine(right)
result = combined.normalize()
assert len(result) == 1
assert 2.0 < result[0] < 4.0
```

For other numerical operations, see `dequant_block_fp8`, `dequant_mxfp4_nibble`, `sinkhorn_birkhoff_substochastic`, `rms_norm`, `linear_cka_consensus`, and `topk_moe_routing`.

## 7. Read metadata and transform streams

### Safetensors header supplied as bytes

```python
import json
import struct
from vld import SafetensorsGlance

header = json.dumps({
    "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
}).encode("utf-8")
prefix = struct.pack("<Q", len(header)) + header

length, fields = SafetensorsGlance.parse_header_bytes(prefix)
assert length == len(header)
assert fields["weight"]["shape"] == [2]
```

`parse_header_bytes` accepts a complete supplied header prefix; callers decide how to obtain those bytes.

### Git packet lines and object reachability

```python
from vld import GitWireV2Parser

raw = GitWireV2Parser.encode_pkt_line("command=fetch\n")
assert GitWireV2Parser.parse_pkt_lines(raw) == [b"command=fetch\n"]

dag = {"C": ["B"], "B": ["A"], "A": []}
missing = GitWireV2Parser.compute_set_difference({"C"}, {"B"}, dag)
assert missing == {"C"}
```

### Process state and local stream transforms

```python
from vld import HeadlessOS, StreamPathMachine

process = HeadlessOS()
process.write_fd(1, b"hello")
assert process.read_fd(1) == b"hello"

stream = StreamPathMachine()
output, accepted = stream.process_hop("reverse", b"abc")
assert accepted and output == b"cba"
assert len(stream.hop_receipts) == 1
```

## 8. Register a Ground or operator rule

```python
from vld import GroundRegistry

registry = GroundRegistry()
registry.register_ground("mod256", {"type": "cyclic", "modulus": 256})
registry.register_lens("utf8", "utf8")
registry.register_rule("identity", lambda term: None)

assert registry.grounds["mod256"]["modulus"] == 256
assert len(registry.rules) == 1
```

`GroundRegistry.default()` returns a shared process-local registry. Instantiate `GroundRegistry()` for an isolated registry.

For proposed runtime extensions and their acceptance tests, see [Research Goals](RESEARCH_GOALS.md).
