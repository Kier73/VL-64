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
