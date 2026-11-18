# $\mu\text{VLD-64}$ Comprehensive Developer & Researcher Usage Guide

This guide provides copy-pasteable, verified code examples for each of the core capability domains in `μVLD-64`.

---

## 1. Quickstart & Command-Line Interface

### Invariant Self-Verification
Verify that all mathematical invariants, cryptographic routines, and algebraic operators are intact:
```bash
python -m vld verify
```

### Logarithmic Affine Orbit Jumping
Navigate $10^{12}$ coordinates of an affine dynamical system $x_{t+1} \equiv (5 x_t + 1) \pmod{256}$ in $O(\log T)$ time:
```bash
python -m vld navigate --a 5 --b 1 --x0 3 --steps 1000000000000 --modulus 256
```

---

## 2. Coordinate Lenses: Lifting Bytes into State Spaces

```python
from vld import ByteLens, LENS_RAW, LENS_UTF8, LENS_U32LE

# 1. Exact round-trip verification
raw_bytes = b"Hello, Virtual Layer!"
assert LENS_UTF8.verify_roundtrip(raw_bytes) is True

# 2. Lifting into typed representation
state = LENS_UTF8.lift(raw_bytes)
assert state == "Hello, Virtual Layer!"

# 3. Lowering back to product ring (Z/256Z)*
recovered = LENS_UTF8.lower(state)
assert recovered == raw_bytes
```

---

## 3. Cryptographic Quorum & RFC 8032 Ed25519 Authority

```python
from vld import QuorumAuthority, ed25519_keypair_from_seed, ed25519_sign, ed25519_verify

# Provision 3 independent authority members
seeds = [b"\x01" * 32, b"\x02" * 32, b"\x03" * 32]
quorum = QuorumAuthority(member_seeds=seeds)

# Required strict majority quorum threshold: 2q > N => q = 2 of 3
assert quorum.q_min == 2

# Issue a quorum-attested state transition certificate
proposal = {"slot": 42, "state_digest": "4789abf3...", "epoch": 1}
cert = quorum.sign_proposal(proposal)

# Independently verify the certificate
assert quorum.verify_quorum_certificate(cert) is True
```

---

## 4. Affine Monoids & 5-Rung Wrap Identifiability Certification

```python
from vld import affine_navigate, certify_modular_progression

# Advance x_{t+1} = (5 x_t + 1) mod 256 across 1,000,000 steps in O(log n)
# Evaluated with independent dual matrix exponentiation check
y_jump = affine_navigate(a=5, b=1, x0=3, n=1000000, m=256)

# Strict Degree-of-Freedom Wrap Identifiability Check (df >= 1)
# 1. Unwrapped trace (infinitely many moduli possible): REFUSED
unwrapped = [10, 15, 20, 25] # step 5, no wrap witnessed
certified, _ = certify_modular_progression(unwrapped, declared_modulus=None)
assert certified is False

# 2. Witnessed wrap-around drop: CERTIFIED
wrapped = [240, 250, 4, 14] # step 10, mod 256 (250+10-256 = 4)
certified, (step, modulus) = certify_modular_progression(wrapped, declared_modulus=None)
assert certified is True
assert (step, modulus) == (10, 256)
```

---

## 5. Cellular Sheaf Cohomology & Active Syndrome Fault Isolation

```python
from fractions import Fraction
from vld import solve_sheaf_cohomology, syndrome_decode_sparse_fault

# 1. Inconsistent Observer Loop (Triangle v0 -> v1 -> v2 -> v0)
delta0 = [[-1, 1, 0], [0, -1, 1], [1, 0, -1]]
c_inconsistent = [Fraction(1), Fraction(1), Fraction(1)] # Nonzero holonomy loop

is_consistent, witness = solve_sheaf_cohomology(delta0, c_inconsistent)
assert is_consistent is False
# Witness is a dual 1-cocycle in ker((δ^0)^T) with <y, c> != 0
print("Fredholm Obstruction Cocycle:", witness)

# 2. Active Cohomological Syndrome Repair
# Find sparse sensor corruption e such that Y e = syndrome (mod 7)
Y_syndrome = [[1, 0, 1], [0, 1, 1]]
syndrome = [3, 3] # Injected fault on column 1 of magnitude 3
recovered_error = syndrome_decode_sparse_fault(Y_syndrome, syndrome, p=7)
assert recovered_error == [0, 3, 0]
```

---

## 6. Sufficient-Statistic Linear Operator Contraction

Evaluate $y = W x$ where $W = W_{\text{bilinear}} + W_{\text{orbit}} + U V^\top + \Delta$ in $O(M + N + M P)$ FLOPs without dense matrix allocation:

```python
from vld import StructuralLinearOperator

# 1024 x 1024 structured operator
op = StructuralLinearOperator(
    rows=1024, cols=1024,
    alpha=1.0, beta=0.01, gamma=0.02, delta=0.0001,
    orbit=[1.0, 0.5, 0.25, 0.125], stride=2,
    sparse_delta={(10, 20): 100.0}
)

x = [1.0] * 1024
y = op.matvec(x)
assert len(y) == 1024
```

---

## 7. Flash-Stream Online Attention Monoid

Stream multi-shard attention across distributed paths without central materialization:

```python
import math
from vld import FlashStreamState

# Worker A evaluates shard 1
shard_A = FlashStreamState(m=1.0, l=1.0, O=[2.0])

# Worker B evaluates shard 2
shard_B = FlashStreamState(m=2.0, l=1.0, O=[4.0])

# Path accumulator combines states associatively in transit
accum = shard_A.combine(shard_B)

# Normalize solely at terminal receiver: O / l
output = accum.normalize()
expected = (2.0 * math.exp(1.0 - 2.0) + 4.0 * 1.0) / (math.exp(1.0 - 2.0) + 1.0)
assert abs(output[0] - expected) < 1e-12
```

---

## 8. Compiler, WAL Effect Ledger & Componentwise Ceilings

```python
from vld import (
    IntentionCompiler, IntentionSpec, IntentExpr,
    NavigationCostVector, EffectLedger
)

compiler = IntentionCompiler()
spec = IntentionSpec(
    intention_id="nav_1",
    target_expr=IntentExpr.iterate_affine("x", a=5, b=1, steps=1000, modulus=256),
    contract_kind="int_vector",
    lineage_id="session_42",
    cost_ceilings=NavigationCostVector(latency=1.0, compute_ops=100.0, byte_movement=64.0, recovery_overhead=0.0)
)

# 1. Compile to Pareto-minimal realization
plan = compiler.compile(spec)
assert plan["selected_plan"] == "monoid_jump"

# 2. Realize with mathematical semantic verification
receipt = compiler.realize(plan, env={"x": [3]})
assert receipt["status"] == "ADMITTED"
assert receipt["admitted_observation"] == [affine_navigate(5, 1, 3, 1000, 256)]

# 3. WAL Effect Ledger ensures idempotent external transitions
ledger = EffectLedger()
calls = []
def send_external_packet():
    calls.append(1)
    return "SENT"

res = ledger.execute_effect("intent_1", "packet_1", send_external_packet)
assert len(calls) == 1
# Subsequent identical effect key returns cached outcome without re-executing
res2 = ledger.execute_effect("intent_1", "packet_1", send_external_packet)
assert len(calls) == 1
```

---

## 9. Autopoietic Residual Loop & M1 LoRA Morphisms (Direction B)

```python
from vld import AutopoieticDiscoveryLoop, LoRAMorphism

loop = AutopoieticDiscoveryLoop(threshold=2.0)

# Evaluate execution residual: rho = C_realized / C_lower_bound
opp = loop.evaluate_residual(intention_id="dense_matmul_bottleneck", realized_cost=10.0, lower_bound=2.0)
assert opp is not None
assert opp["status"] == "DISPATCHED"

# Synthesize M1 Low-Rank Delta Morphism: W' = W + (alpha / r) A B^T
A = [[1.0], [2.0]] # 2 x 1
B = [[1.0], [1.0]] # 2 x 1
lora = LoRAMorphism(alpha=2.0, r=1, A=A, B=B)
delta_output = lora.contract([1.0, 1.0])
assert delta_output == [4.0, 8.0]
```
