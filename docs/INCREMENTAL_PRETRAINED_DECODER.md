# Incremental 30-layer causal decoding across recoverable model-orbit heads

This follows the validated 30-layer layer-major prefill and full-vocabulary
projection. `incremental_decoder.py` composes **one newly admitted token** with
all exact, content-addressed K/V pages from a previously completed decoder
checkpoint. It does not replay prior token positions.

## Evidence-preserving procedure

1. A trusted caller supplies the **expected checkpoint digest** independently
   of checkpoint bytes, plus a strictly higher owner epoch. The complete
   previous checkpoint, hidden bytes and **all 30** K/V histories are verified
   before requesting any new weight bytes.
2. The new token embedding is admitted through the existing bounded
   `QualifiedTensorReader`, with source-host and exact SHA-256 witnesses.
3. Each layer reacquires its declared nine tensors, verifies the decoded
   **static weight-content identity** against the previous checkpoint, loads
   that layer's old causal K/V prefix, and computes only the new token's
   RoPE/GQA/MLP path using the existing `PreparedLayer`.
4. Each layer produces a new content-addressed K/V page manifest of length
   `old_prefix + 1`. A failed stage leaves the last committed checkpoint and
   the in-memory accepted prefix unchanged; unreferenced new pages are inert.
5. Only after all layers succeed is the new token appended to the committed
   prefix. A new F64 hidden-state page manifest and causal head are published.
   Full model RMSNorm and all vocabulary logits then run through the existing
   bounded final-projection path.

Different **qualified weight-source hosts** can supply matching static tensor
bytes. That changes transport provenance without changing the mathematical
model. This is an observer-host data-rehoming contract, not yet verified remote
processor execution. Digests provide integrity relative to an external pin;
a durable multi-writer authority requires E-CPU quorum fencing/admission.

## Numerical and recovery tests

The mandatory `tests/test_incremental_decoder.py` independently verifies:

- A 30-layer BF16 and F32 next-token step matches an entire fresh causal
  reference model forward pass for the three-token sequence.
- Different source-host identities preserve output when admitted static
  weights are equal; previous positions are not re-executed.
- A second handoff to another owner epoch and another appended token still
  matches an independent four-token reference.
- Changing a finite BF16 static weight and recomputing its per-range hash is
  insufficient to satisfy the externally pinned weight identity.
- Stale epoch, incorrect checkpoint digest and incomplete model position are
  refused before the continuation can be published.

This changes **no files** under `src/vld/` and remains an adapter above the
65,093-byte compact runtime.

## Example

```python
from experiments.model_fabric.incremental_decoder import IncrementalDecoder

# checkpoint is the admitted complete 30-layer checkpoint of existing tokens.
# expected_head comes from a separate trusted checkpoint authority.
step = IncrementalDecoder(
    plan, checkpoint, expected_head=expected_head,
    host_for=observer_for_layer,
    approved_for=policy_for_layer,
    store=paged_store, owner="next-qualified-observer", epoch=checkpoint["epoch"]+1)
step.append_token(previous_logits_receipt["selected_next_token"])
next_logits = step.logits()
new_checkpoint = step.checkpoint()
```

A tokenizer, sampler and a fully independently verified **real pretrained**
autoregressive sequence are follow-up experiments. Do not describe tests on
synthetic fixtures as observations of real pretrained next-token generation.
