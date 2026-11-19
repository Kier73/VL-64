# Thirty-layer pretrained decoder: bounded execution and exact state recovery

This model-fabric extension follows the earlier typed DAG, real two-layer
SmolLM2 execution, E-CPU HFG-003/M1 state handoffs, and VLD's paged KV orbit.
It adds **no code to `src/vld/`** and preserves the 65,093-byte VL-64 core.

## Execution contract

`full_decoder.py` takes the already admitted `vl64.typed-pretrained-execution-dag.v1`.
For every layer, it checks all typed operator dependencies with the existing
`_verify_layer_nodes`, obtains nine observed weight tensors through
`QualifiedTensorReader`, and executes every supplied causal token position via
`PreparedLayer`. Only a single layer's decoded weights are held concurrently;
after its full output and KV state are independently published to the page
store, the next layer begins. Each input token embedding is read only once.

The complete model output additionally validates the final RMSNorm and
vocabulary-projection nodes. The selected vocabulary projection uses the
**observed** weight binding, including tied embedding weights when declared.
All vocabulary rows are processed in bounded windows, rather than fabricating
unread weights or truncating the output vocabulary. The receipt includes the
argmax token, top logits, SHA-256 of all F64 logits, source range evidence,
model/revision/DAG digests, and full layer count.

For `HuggingFaceTB/SmolLM2-135M` pinned at revision
`4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`, the observed manifest
contains **30** decoder layers, **272** BF16 tensor declarations, a **576**
dimensional hidden state, **nine** query heads, **three** KV heads, and
**49,152** vocabulary rows. The core remains below the 64 KiB seed size;
physical page/range transactions are separately bounded to 64 KiB.

## Recovery lineage

The extension reuses `ContentAddressedPages` from the E-CPU/VLD-derived orbit
adapter. After a complete layer has been executed, it records exact F64 hidden
bytes, that layer's F64 causal K/V histories, a static weight-content identity,
range-witness chain, and a causal transition digest. An atomic root checkpoint
binds the exact token prefix, model/revision/DAG, completed layer coordinate,
all already committed KV page addresses, weight identities and epoch/owner.

Restoring at the next layer requires an **externally supplied expected checkpoint
head** and a strictly higher epoch. The checkpoint's digest is not itself an
authority grant: trusted caller and byte transport policies are separate. The
fresh process verifies every saved KV page and hidden-state extent before it
requests any weight range from the new host. Successfully committed earlier
layers are **not re-executed**. Failure during a layer does not publish a new
root checkpoint or advance the committed layer coordinate.

This establishes single-writer process continuation and source reassignment,
not distributed execution admission, authenticated remote processor work, or
quorum fencing across simultaneous writers. Those remain distinct E-CPU
integration paths.

## Independent verification

`tests/test_full_decoder.py` compares **30 entire synthetic decoder layers**
and every final vocabulary logit against an independently written scalar
reference, for BF16/F32 weights and tied/untied embedding heads. It checks
independent NumPy per-layer results, a simulated process boundary after layer
17, exact logit preservation, corrupt KV refusal, stale/incorrect checkpoint
heads, incomplete model refusal, and atomic failure semantics.

`full_decoder_lab.py` adds a separate optional **actual pretrained** experiment.
It reads exact pinned BF16 tensor ranges through the existing Hugging Face
host and calculates each real layer again with a separate dense NumPy oracle.
Unlike the core execution, the oracle does not call VL-64's
`StructuralLinearOperator`, `rms_norm`, RoPE, attention or SwiGLU helpers.
Each layer checks every resulting hidden coordinate. The full projection
independently checks all 49,152 logits and top-token identity. An unavailable
range or numerical discrepancy fails the run with `UNRESOLVED` evidence.

The optional GitHub Actions live job uses **two distinct Python processes**:
the first computes layers 0 through 14 and publishes a checkpoint head to a
GitHub Actions step output; the second uses that externally carried head with
an increased owner epoch, loads exact pages, and computes layers 15 through 29
and the complete vocabulary. This is process transfer, not a second machine.
The fixture tests are mandatory; the optional public-network experiment is
reported separately rather than claiming a real pretrained result from a
synthetic fixture.

## Commands

```bash
python -m pip install -e '.[dev]'
python -m pip install numpy
python -m pytest -q tests/test_execution_dag.py tests/test_pretrained_layer.py tests/test_pretrained_chain.py tests/test_pretrained_orbit.py tests/test_full_decoder.py

# Optional: requires outbound HTTPS and honored byte range requests.
python -m experiments.model_fabric.full_decoder_lab \
  --tokens 42 75 --through-layer 15 --owner first-process --epoch 0 \
  --state-dir artifacts/full-pretrained \
  --result-out artifacts/pretrained-partial.json

# Copy the printed VL64_CHECKPOINT_HEAD digest from the first command.
python -m experiments.model_fabric.full_decoder_lab \
  --tokens 42 75 --through-layer 30 --owner second-process --epoch 1 \
  --resume-from artifacts/full-pretrained/checkpoint.json \
  --expected-head '<externally-recorded-first-process-head>' \
  --state-dir artifacts/full-pretrained \
  --result-out artifacts/pretrained-complete.json
```

All real-model outputs are **forward passes for already supplied token IDs**.
They do not establish the tokenizer's text-to-ID semantics, a sampling policy,
cryptographic authentication of the entire model checkpoint, or true compute
execution across separate remote processors. These are the next independently
testable integration points after a successful 30-layer run.
