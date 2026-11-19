# Complete pretrained layer as a navigable Ground

This follows commit `6389481` (portable real model slice), the typed full
SmolLM2 manifest DAG (PR #18), and the genuine pretrained token embedding/Q
probe (PR #19). It deliberately introduces no new code in `src/vld/`.

## What is now executable

`pretrained_layer.py` interprets the already compiled
`vl64.typed-pretrained-execution-dag.v1`, checks the exact named stage
operator types, and acquires all nine required first-layer tensors from
approved observer-hosts with exact byte-range witnesses:

1. Token embedding row from the pinned Safetensors tensor offsets.
2. Input RMSNorm, actual Q/K/V projections, and position-dependent Llama RoPE.
3. Causal grouped-query attention with nine query heads and three shared
   key/value heads, retaining rotated keys and values in temporal state.
4. Attention output projection, first residual, post-attention RMSNorm.
5. Real gate/up projections, SwiGLU, down projection, second residual.

The full first layer has 3,540,096 BF16 parameter values (7,080,192 bytes).
The pinned model's embedding row adds 1,152 bytes per token. These are
communication-byte counts, not the size of the VL64 core, not a claim of
compression, and not a full model download. The range message cap is 65,536
bytes and the nine tensors need a minimum of 113 nonempty range messages
when acquired separately. Two distinct token embeddings add two messages.

## Numerical independence

`real_layer_lab.py` reconstructs values only from exact range payloads the
host actually returned, then computes the layer again with an **independent
dense scalar implementation** using `math.fsum`: no VL64 linear operator,
RMSNorm, RoPE or GQA helpers are used in that reference. The test harness
checks both BF16 and F32 fixture weights, causal position changes, weight
reuse, SHA-256 witness rejection, and rejection of a recomputed DAG digest
with an unadmitted operator. The optional network run uses pinned
`HuggingFaceTB/SmolLM2-135M`, revision
`4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`, and token IDs
`[42,75]`. It compares every coordinate of the final 576-element output
within absolute tolerance 2e-5.

```bash
python -m pytest -q tests/test_execution_dag.py tests/test_pretrained_layer.py
python -m experiments.model_fabric.real_layer_lab \
  --out artifacts/pretrained-layer0-report.json
```

The live command requires outbound HTTPS and strict HTTP 206 Range support.
If the remote server refuses, the experiment creates an `UNRESOLVED`
receipt and exits nonzero. The CI workflow retains the evidence JSON whether
the remote model is available or not, while keeping fixture tests mandatory.

## Trust and scope

A digest bound to each accepted byte range detects accidental or adversarial
changes **relative to the returned bytes**; it is not an independent signature
of the entire Hugging Face checkpoint. Host authorization is an explicit
caller-approved set. The pinned commit and HTTPS transport constrain the
expected source, but full-checkpoint authenticity requires further evidence.
A host can be a remote byte provider, a cache, or an observer capable of
satisfying the exact typed request; new computational receipts must still
pass independently defined numerical contracts.

The layer output is a pretrained *intermediate hidden state*, not language
generation or vocabulary logits. A full-model claim requires executing all
30 transformer layers, final normalization, complete vocabulary projection,
and validation against an independently implemented reference with the
same tokenization and numerical semantics.

## Next composition

* Reuse the prepared layer's KV cache across multiple accepted token positions.
* Compose 30 qualified layer Grounds with explicit handoff receipts, avoiding
  re-fetches of already admitted tensors where durable cached weight evidence
  exists.
* Add row-partitioned output projection, batching, and variable observer
  assignment; verify that a replacement observer preserves both weight
  identity and the typed result under the same numerical tolerance.
* Only then integrate independent tokenizer and language-token generation
  tests. Keep the `src/vld/` 64 KiB core ceiling unchanged.


## Observed result — first full pretrained layer

The first network experiment completed successfully on GitHub Actions:
[run #44451577911](https://github.com/Kier73/VL-64/actions/runs/44451577911).
The real pinned model admitted **7,082,496 BF16 bytes** from **115**
bounded HTTP range requests: 7,080,192 bytes for all nine layer-0
tensors plus 1,152 bytes each for token IDs 42 and 75. Both tokens were
passed through the same prepared transformer layer with causal KV state.

The reference independently computed all 576 output coordinates, reporting
max absolute error **0.0 when rounded to 12 decimal places**
(tolerance 2e-5). The first eight actual layer outputs are
`[-0.00034656, 1.22291947, 1.55508065, -4.18592866, -0.1411358,
-0.18874785, -0.63662119, -0.67673943]`.

The exact emitted report is preserved at
[`datasets/2026-11-18-real-layer0-reference-receipt.json`](../experiments/model_fabric/datasets/2026-11-18-real-layer0-reference-receipt.json).
Its SHA-256 is
`97f7b3c63b9372cc5576e3f0cf03d68bee5d36c5f53de2d18994b9ee92483da9`,
and `tests/test_saved_real_pretrained_layer.py` verifies the stored
report's bytes and model/tensor-plan accounting without network access.

This evidence supports exactly **one complete pretrained transformer layer**
with real BF16 weights and independent scalar numerical agreement. It does
not establish full-model generation, a cryptographic whole-checkpoint
authenticity proof, or observer execution across independent physical hosts.
