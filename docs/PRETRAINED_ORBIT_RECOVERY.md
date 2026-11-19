# Recoverable pretrained model orbit — cross-repository evidence path

This experiment continues the [typed pretrained execution DAG](TYPED_PRETRAINED_EXECUTION_DAG.md), [verified layer](PRETRAINED_LAYER0_GROUND.md), and [two-layer chain](PRETRAINED_LAYER_CHAIN.md) without adding bytes to `src/vld/` (65,093 bytes at implementation).

## Previously implemented machinery examined

The approach incorporates **contracts**, not a copy of whole infrastructure:

- [E-CPU `model_orbit_handoff.py`](https://github.com/Kier73/E-CPU/blob/main/mailcpu/model_orbit_handoff.py) (HFG-003) persists an exact layer output as an M1 content-addressed object, and requires a **separately trusted expected state digest** on recovery. Merely hashing a received object does not authorize its interpretation.
- [E-CPU `navigable_computation_checkpoint.py`](https://github.com/Kier73/E-CPU/blob/main/mailcpu/navigable_computation_checkpoint.py) binds the restored DAG frontier, retry identity, route evidence and receiver state. [E-CPU `tensor_checkpoint.py`](https://github.com/Kier73/E-CPU/blob/main/mailcpu/tensor_checkpoint.py) provides single-writer atomic file replacement. Full multi-writer reassignment requires independent quorum fencing, not a field containing an epoch.
- [Virtual Layer Dynamics `models/orbit.py`](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/models/orbit.py) already carries paged KV-state Grounds and crash-resumable model checkpoints. Its refusal to pretend a causal softmax history is an exact finite affine summary is retained here: **every needed K/V coordinate is preserved**.
- [Virtual Layer Dynamics `authority/temporal.py`](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/authority/temporal.py) makes scheduler-scoped causal coordinates and epochs explicit. Here, `(owner, epoch, token position)` is a **caller-controlled local assignment coordinate**, not cross-machine time consensus.

The E-CPU and Virtual Layer Dynamics modules are not imported: both maintain their own independent infrastructures and `vld` package names. The reusable bridge is the precise behavioral contract so a future authenticated adapter can map onto either fabric without introducing a second authority or scheduler.

## Added experiment

`experiments/model_fabric/pretrained_orbit.py` adds a `RecoverableModelOrbit` around the existing `PreparedLayer` and `QualifiedTensorReader`:

1. Advance **one token at a time**, through layers 0 and 1, using the exact typed DAG. Each successful token commits causal K/V frames for both layers, the previous handoff digest, transcript head, token prefix and final hidden vector.
2. Encode K/V frames and final hidden coordinates as **little-endian F64**, identical to the Python intermediate-value representation; no lossy F32 checkpoint conversion. Split every byte object into SHA-256 addressed pages no larger than **65,536 bytes** and persist them with atomic replacement. Layer-state manifests are separate addressed pages and the root checkpoint fits in one bounded frame.
3. Publish one canonical checkpoint with the pinned model/revision, DAG and tensor-ground identity, full token prefix, active assignment owner/epoch, root manifests, observed weight witness chains, causal handoff head, and predecessor checkpoint head.
4. To restore, require an **expected checkpoint head supplied from outside the untrusted checkpoint**, reject stale assignment epochs, verify the root, all layer manifests, every K/V byte page, hidden-state bytes and causal prefix lengths **before** acquiring any new weight bytes. Then recreate the PreparedLayers with independently approved range providers, independently recompute each layer's exact F64 static-weight content identity, and refuse changed numerical weights before recovering K/V history, without replaying the previous token positions.
5. A failed new token/transport/state write does not publish a new frontier: all layer K/V state, read-witness chains, counters and the token prefix roll back to the prior committed position. Newly written but unreferenced pages are harmless; their garbage collection remains a separate storage concern.

`ContentAddressedPages` and checkpoint JSON use fsynced writes plus atomic filesystem replacement. This is for **single-writer local crash recovery** only; it is not a substitute for E-CPU's quorum fencing or process-to-process mTLS. A restored session requires a caller-supplied greater epoch and an explicitly approved new host. Host names are **attribution labels** and not cryptographic attestations. The host-independent SHA-256 static-weight identity is computed from each admitted tensor's exact decoded F64 values, so an approved new host cannot silently replace the predecessor's weights with different self-consistent bytes. Real hosted tensor execution still occurs inside the calling Python interpreter; range hosts provide exact pretrained bytes.

### Test and optional live experiment

```bash
python -m pytest -q tests/test_execution_dag.py tests/test_pretrained_layer.py \
  tests/test_pretrained_chain.py tests/test_pretrained_orbit.py

# Optional, reads the actual pinned model ranges; no whole-checkpoint fetch.
python -m experiments.model_fabric.real_orbit_lab \
  --out artifacts/pretrained-orbit-report.json \
  --state-dir artifacts/pretrained-orbit-pages
```

The independent fixture suite compares **restored** BF16/F32 decoder results against a separate scalar reference. It also launches an actual **second Python interpreter** that loads the previous interpreter's KV pages and checkpoint from disk, continues token position 2 under a distinct host attribution, and checks the final result. Tests explicitly refuse stale/recomputed checkpoint roots, altered/missing pages, wrong model identity, replayed positions, unqualified replacement hosts, nonfinite restored values, and mid-commit failures.

`real_orbit_lab.py` is a separate optional public HTTPS experiment for pinned `HuggingFaceTB/SmolLM2-135M` revision `4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`: it runs token 42, saves a complete real-BF16 checkpoint, restores to a new *logical* observer assignment, runs token 75 without replaying the earlier step, and checks the final hidden state against an independent dense scalar implementation reconstructed from returned BF16 ranges. Two logical source IDs use the same trusted pinned HTTPS endpoint; they do **not** establish two physical host processors or a separate-process live model trial. On any network/range/math failure, the experiment exits **UNRESOLVED**, never a fabricated success.

## What this enables next

- Bind checkpoint publication to E-CPU's **authenticated quorum high-water record and fencing term** to prevent two independent writers from both publishing different valid heads.
- Separate the currently local layer compute into distinct **receiver processes** using E-CPU's authenticated path/reliable channel and exact typed output receipts. Rehome a *compute* host, not just the weights' provider.
- Retain admitted weight-cache identities as durable Grounds to avoid re-fetching complete layer matrices after each process restart; continue retaining independently verified model values.
- Compose the same verified state snapshots through additional pretrained layers toward all 30, then final normalization, vocabulary logits and tokenizer verification. This experiment remains **two-layer partial model computation**, not full language generation.

### Checkpoint schema upgrade after initial recovery evidence

The first restart prototype (PR #22) used `vl64.model-orbit-checkpoint.v1` without
a persistent static-weight content identity. The strengthened format is
`vl64.model-orbit-checkpoint.v2`. It **does not silently import v1 checkpoints**
into the stronger trust contract: doing so would equate self-hashed range
receipts with a trusted pre-restart weight snapshot. Re-admit the prior weight
Ground through an independently trusted source and publish a new v2 state,
or continue evaluating the original v1 checkpoint using the pinned original
implementation. Never substitute a changed v1 snapshot's supplied digest for
an externally authorized high-water head.
