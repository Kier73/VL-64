# MPF-002 — Source-pinned observer sessions with host-resident trained behavior

## Goal and data movement

MPF-001 produces original E-CPU HSS-002 model source Grounds and independently
recoverable VL-64 observers of their static **structure**. MPF-002 binds these
sessions to pre-existing **qualified numerical behavior** via VL-64's original
`operational_ground.py`, `operational_hosts.py`, and `intention_fabric.py`.

The static model remains resident in its source/execution host. The receiver
communicates only source/host identities, bounded prompt/token inputs, F32
logit tiles, output text and hashed causal receipts. A public model page alone
provides **metadata**, not execution access: a callable numerical host and
independently supplied artifact identity are mandatory for behavioral admission.

The bridge does not perform HTTP model downloads, weight-range reads,
checkpoint loading or tokenizer downloads. Runtime-specific resident hosts
are injected by the caller.

## Executable contract

For external source `m`, original source Ground `G_m`, receiver observer `o`,
and source-pinned current observer tip `L_{m,o}`:

```text
source = E-CPU HSS-002 SourceStructure(source repository, immutable revision)
operator = observe_operational_ground(host,
    expected_model=source.repository,
    expected_revision=source.revision,
    expected_artifact=independently_pinned_artifact,
    approved_hosts=approved_host_ids)

behavioral_genesis = H("vl64.source-pinned-behavior-fleet.v1.genesis",
    observer, alias, source_head, source_tip, epoch, operator.ground_digest)

turn:
    assert source_observer.tip == bound_source_tip
    intent = ObserverIntention("model.behavior.generate",
        "hf://repository@revision", bounded conversation options)
    compiled = compile_intention(operator, kind, full_dialogue, budget)
    observation = IntentionSession(compiled, operator, host).run()
    checkpoint = navigator.checkpoint()  # token prefix + numerical commitments
    behavioral_tip = H("vl64.source-pinned-behavior-fleet.v1.receipt",
        old_tip, source_head, source_tip, operator_ground,
        intent, observation, checkpoint, observer, epoch, sequence)
```

The original numerical execution includes `encode -> open ->
logits_tile (complete F32 vocabulary coverage) -> advance ->
decode`. VL-64 recomputes tile SHA-256, checks full logits commitment,
and chooses argmax independently of the host's reported choice.
Each accepted observer turn has its own high-water receipt and explicit
idempotency key. Repeated requests return their earlier receipt.

## Shared original model, many observers

```python
from experiments.model_fabric.source_pinned_behavior_fleet import SourcePinnedBehaviorFleet

# `pages` is an already populated VirtualModelPageFleet.
# `resident` is an existing qualified callable at the source model host.
# `artifact_sha256` comes from independent admission, not resident.describe().
behavior = SourcePinnedBehaviorFleet(pages)
behavior.bind("qwen", resident, approved_hosts={"qwen-inference-host"},
              expected_artifact=artifact_sha256)
for observer in ("researcher-1", "researcher-2"):
    pages.open(observer, "qwen", host_id="vl64-receiver")
    behavior.open(observer)

receipts = behavior.turn_many([
    {"observer_id": "researcher-1", "user_text": "Describe state A.",
     "request_id": "a1", "max_new_tokens": 8},
    {"observer_id": "researcher-2", "user_text": "Describe state B.",
     "request_id": "b1", "max_new_tokens": 8},
], workers=2)
assert receipts[0]["source_head"] == receipts[1]["source_head"]
assert receipts[0]["head"] != receipts[1]["head"]
assert all(r["model_payload_bytes"] == 0 for r in receipts)
```

A shared mutable resident numerical host is locked across a complete
conversation's numerical operations. This prevents two observers from
crossing host session/KV-cache state or evicting each other's logits
halfway through a causal sequence. Multiple different source-host
objects can infer in parallel. **Parallel observer scheduling over
one serialized host is not claimed to be simultaneous GPU kernel execution**.
The future adapter can improve host-side multiplexing with independently
scoped KV and per-session quotas while retaining these causal pins.

## Portable recovery and rehome

`snapshot()` stores the MPF-001 source snapshot, admitted operator
Grounds, open-session anchors, bounded dialogue history, individual turn
events, and numerical checkpoints. `restore(...)` independently replays
all behavioral hashes, source pins, turn-intention correspondence, operator
and checkpoint digests; it never calls the numerical host. It cannot
mathematically prove that an untrusted remote host used its advertised
original weights. The execution trust and original artifact checks belong
to the host's authenticated admission layer.

On replacing or reattaching a trained host, `bind()` challenges the latest committed token prefix of each affected observer and requires **byte-identical F32 full-logit commitments** from the candidate host. A host merely reusing the source revision and artifact identifiers but producing altered logits is refused. This is a per-observed-prefix numerical identity check, not an all-input proof or cryptographic remote attestation.

After a verified `pages.rehome()`, the behavior observer must
`reanchor(observer, expected_tip=old_behavior_tip)` before it can
publish another turn. A recovered bridge must call `bind` with the
same original host/revision/artifact Ground before continuing.

## Benchmark and exact evidence category

Run the first real *host-resident trained-operator* integration locally
(without model page payload or network access):

```bash
python -m pytest -q tests/test_source_pinned_behavior_fleet.py
python -m experiments.model_fabric.source_pinned_behavior_fleet_lab \
  --observers 16 --workers 8 --tokens 3 \
  --out artifacts/mpf002-trained-host-fleet.json
```

The benchmark uses **one actually trained count-based byte language model**
resident in the host fixture and evaluates its learned sparse weights
through `vld.StructuralLinearOperator.matvec`. For an honest test of
original-host identity handling, it uses an explicit **synthetic**
HSS-002-compatible model source identity. It exercises real numerical
execution, complete logit audit, observer isolation, replay, host-call
accounting and zero weight copies. It is **not** a claim of executing
the actual DeepSeek/Qwen/SmolLM2 public pretrained weights.

### First verified trained-behavior benchmark (GitHub Actions Linux 3.11)

[Successful measured job](https://github.com/Kier73/VL-64/actions/runs/45104139643). One trained byte-LM host, one synthetic source Ground, bounded three-token causal conversations, eight scheduling workers:

| Observers | Independent causal receipts | Computed tokens | Host protocol calls | Total execution wall time | Payload bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 3 | 9 | 2.71 ms | 0 |
| 8 | 8 | 24 | 72 | 17.74 ms | 0 |
| 16 | 16 | 48 | 144 | 34.17 ms | 0 |

The 16-observer checkpoint reconstructed in approximately 7.29 ms on the
same CI worker. Results describe the local, serialized trained host fixture,
not GPU inference throughput of a public model or its remote serving latency.

The cross-platform workflow
`.github/workflows/source-pinned-behavior-fleet.yml` also runs the
original VL-64 `operational_ground`, `intention_fabric`, and
`virtual_page_fleet` regression suites.

## Next experiment: actual remote pretrained original host

Connect an authenticated, source-resident pretrained transformer or
inference endpoint implementing the same `CALL_SCHEMA` contract, with a
caller-verified checkpoint artifact identity, source revision and allowed
host identity. Measure end-to-end original-model behavior across 1, 8,
64, 128 observers, source-host CPU/GPU time, serialized versus
multiplexed inference latency, per-session KV pressure, and cross-host
reassignment challenges. Independently compare selected causal traces
to another qualified implementation at the same immutable revision.
Original pretrained behavior is a separate evidence category from both
MPF-001 structural interpretation and this trained numerical fixture.
