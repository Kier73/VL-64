# VL-64 typed pretrained tensor execution DAG

This extension builds on [commit 6389481](https://github.com/Kier73/VL-64/commit/63894819a9b6df9f6a38f9e79cbaa5ed12832f84), which independently replayed a genuine 5,760-byte BF16 SmolLM2 subgraph. The existing `src/vld/` source and independently reproducible Python model seed are unchanged.

## Reproduce structure, without acquiring any model weights

```bash
python -m experiments.model_fabric.model_dag_lab
python -m pytest -q tests/test_execution_dag.py
```

The saved pinned `HuggingFaceTB/SmolLM2-135M` Ground describes 272 tensors: 30 Llama decoder blocks, hidden width 576, nine query heads, three KV heads, head dimension 64, intermediate width 1536 and 49,152 vocabulary rows. Compiling emits 454 ordered typed nodes, including a token input, embedding, 15 computational nodes per block, final RMSNorm and vocabulary projection.

Compilation checks both observed Ground digests, typed tensor extents and shape compatibility, GQA divisibility, RoPE parity, supported activation, and tied versus independent projection. An observed manifest is **not** authorization to execute weight-backed inference, and a compiled graph is not a computed model output.

## Typed execution stages

`embedding.lookup` selects exactly the externally supplied token ID row. Each layer performs `rms_norm → q/k/v linear → q/k RoPE → causal GQA (with per-layer temporal KV history) → o_proj → residual → rms_norm → gate/up linear → SiLU(gate)*up → down_proj → residual`. Final `rms_norm → vocabulary linear → argmax` yields logits and a greedy next-token ID.

RoPE follows Llama's **split-half** rotation, not interleaved pairing. GQA repeats each KV head logically across `query_heads / kv_heads` query heads, preserving the causal prefix. The final LM head reads `model.embed_tokens.weight` when `tie_word_embeddings=true`; otherwise `lm_head.weight` is required. No generated text claim is made without supplying actual tokenizer/decoding semantics.

## Qualified observer-host ranges

The caller supplies a trusted host allowlist and a callback `host(request) → receipt`. Each request includes pinned model identity, revision, tensor-Ground digest, tensor name, dtype and absolute byte start/end. Each result must include exact payload bytes (16 KiB maximum per transfer), all matching request coordinates, allowed host identity, `QUALIFIED_EXACT_RANGE` status, and the actual SHA-256 of the bytes. F32/BF16 decoding rejects nonfinite values. Observed range hashes are folded into a domain-separated transcript chain.

Ranged access is **not** full-model download: token embedding fetches one row, while matrix/vector multiplication processes bounded row blocks using VL-64 `StructuralLinearOperator`. The optional `huggingface_range_host` uses the project's existing strict HTTPS 206 Content-Range reader and pinned revision URL. An observer-host attestation and per-range self-hash bind a local execution transcript; they do **not independently authenticate** the complete original checkpoint. For stronger third-party provenance, use a separately trusted whole-file digest or authenticated Merkle membership proofs for all ranges.

## Actual remote execution is opt-in

```bash
# Potentially thousands of small HTTP ranges; no tokenizer is included:
python -m experiments.model_fabric.model_dag_lab --tokens 42 --allow-remote-weights
```

The opt-in computation requires every requested byte range to arrive successfully and passes no skipped/unknown weights. It may be slow because this first implementation uses pure-Python VL-64 operators and no persistent cross-token tensor cache. For reproducible stage-level verification the tests use deterministic **synthetic** F32/BF16 byte artifacts and an independently implemented complete scalar decoder; this must not be confused with pretrained text generation. The pinned real-model manifest test verifies structure without asserting real pretrained full-graph numerical inference.

## Next experiments

Stream and cache qualified ranges across token steps, check weight provenance against a publisher-signed content digest, implement reusable KV-cache checkpoints and resumable host rehoming receipts, integrate external tokenization, and compare complete real-model logits against an independently executed reference. Architecture-specific compilers beyond this strictly checked Llama adapter can be added without altering the 64 KiB interpreter core.
