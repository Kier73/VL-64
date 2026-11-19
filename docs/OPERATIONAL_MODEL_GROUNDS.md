# Operational Model Grounds: VL-64 as a virtual processor of host-resident models

## Architectural correction

**The Virtual Layer should not need to store a copy of every trained weight.** A qualified observation can establish an *operational Ground* — a portable, typed description of an observer-host's admitted computational relations and the coordinates used to navigate them. The host retains the actual parameters, and the Virtual Layer interprets requested computation, sends bounded activation/token frames, and checks the resulting observations.

Two complementary execution patterns are implemented, outside `src/vld/`:

1. **Token-model orbit** (`operational_ground.py`, `operational_hosts.py`): observe a model identity, tokenizer, vocabulary, causal context, and the host's supported token operations. Ask the host to encode a prompt, prepare a causal state, and project *all* next-token logits in bounded 4,096-element F32 tiles. VL-64 checks full-vocabulary coverage and the host's total logit commitment, independently computes argmax, advances the token prefix, and records a chained receipt. The model's weights never cross this interface.
2. **Procedural operator graph** (`operational_tensor_graph.py`): observe qualified host-resident matrix operators with exact shapes and claimed weight digests. Compile a typed acyclic Ground, then interpret `input`, `linear`, `silu`, `multiply`, and `add` stages. Each trained matrix-vector product executes at the observer-host; only bounded F32 *activation* tiles cross the Virtual Layer. Nonlinear and residual operations compose inside the interpreter. Arbitrarily wide matrix inputs and outputs are tiled; no weight tensor is sent to the Virtual Layer.

The operator graph is the stronger distinction from a conventional full-model API: VL-64 chooses and interprets the mathematical sequence while hosts supply *parameter-dependent operations*. The token-model interface is useful for complete language benchmarks while mathematical stages of newer architectures become available through the operator graph.

## Tested model participants

**Trained byte-language host.** `HostedTrainedByteLanguage` wraps VL-64's existing actual count-trained byte model. Its weights are held only by the host. The operational navigator runs real trained next-byte generation from the Ground and is checked against an independent trained-count oracle. This is a first-order byte model, **not** an LLM or a conversational agent of GPT-like capability. The local benchmark demonstrates a natural-language text input/output path but is not a claim of high-quality reasoning.

**Optional pretrained transformer host.** `HostedTransformersLanguage` can load an immutable pinned Hugging Face causal model under PyTorch/Transformers with `trust_remote_code=False`, provide native tokenization, and execute pretrained forward passes inside the host. VL-64 still receives only bounded logits and operation receipts. The optional live benchmark pins `HuggingFaceTB/SmolLM2-135M` at `4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`. This is an implementation path, **not** an independently successful live transformer observation until a real model-host run completes.

**Procedural learned-operator fixture.** A host stores two actual numerical matrices and VL-64 composes a typed linear → SiLU → linear → residual computation across those weights, checked against an independent scalar reference. Tests additionally stream 4,100-row projections, reinstantiate the same Ground in another Python process, refuse bad graph edges, and refuse forged output windows. This is not a claim of Qwen or DeepSeek's full attention computation.

## Evidence and recovery

- **No weight caching inside VL.** The Ground stores metadata, model identity, tensor operator identity, shapes, typed DAG edges, and hashes. Trained weights remain with hosts.
- **64 KiB transport ceiling.** Tokens and activations are sent as bounded F32 windows. Model logits are observed at ≤4,096 F32 values per tile (16,384 raw bytes). Tensor input and output channels use the same bound. Host computation and storage may be much larger than a frame.
- **Complete logit selection.** The navigator hashes all tiles in order and verifies they reconstruct the host's committed full logits before selecting argmax (lowest index breaks ties). It verifies completeness of the *returned* result, not independent numerical correctness of the underlying model.
- **Recoverable causal Ground.** The checkpoint records the exact token prefix, cumulative transcript, host-reported full-logit digest, owner, and epoch. Restart with an externally supplied expected checkpoint digest forces a new host process to recreate or restore its causal state and reproduce the committed next-step logits. No model weights are serialized into the checkpoint.
- **Observer rehoming.** An alternate logical host identity can be admitted with the same independently expected model, revision and model-artifact claim. A challenge compares its exact F32 next-logit fingerprint at the checkpointed prefix before accepting reassignment. This **does not prove complete model equivalence** beyond the tested prefix.
- **Model agent tasks.** `BoundedTaskAgent` allows generated strict JSON to request only *caller-registered* tools. The model's output is untrusted. The controlled test exercises an allowlisted arithmetic task and rejects an unauthorized shell request. No uncontrolled tool execution is provided.

## Interpretation and limits

An arbitrary model's trained numerical weights cannot be recreated *exactly* from a shorter metadata description unless a genuine structure/algorithm provides that compression or an equivalent source of information remains accessible. This experiment realizes the other option: a host-resident trained operation is accessible as a typed relation through the Virtual Layer. The model host still performs physical computation; the VL directs, composes, measures and verifies the observations it can actually check.

A host-supplied SHA-256 and observer name are **not cryptographic remote attestation**. The graph does not prove that a hostile host implemented the promised matrix; the logit navigator cannot prove that the host's logits arose from its advertised weights. Authentic channels, separately trusted model-artifact pins, independent numerical references or proofs, and quorum-fenced execution are needed for stronger claims. Here, logit completeness and numerical selection are verified; transformer internals are not independently proven solely by these receipts.

## Reproduction

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_operational_ground.py tests/test_operational_tensor_graph.py tests/test_operational_model_lab.py
python -m experiments.model_fabric.operational_model_lab \
  --backend byte --prompt 'Virtual Layer' --steps 8 \
  --out artifacts/operational-byte-language.json
```

Optional real pretrained run requiring PyTorch/Transformers, outbound access and model-host capacity:

```bash
python -m pip install torch 'transformers>=4.50,<5'
python -m experiments.model_fabric.operational_model_lab \
  --backend transformer --prompt 'The capital of France is' --steps 3 \
  --out artifacts/operational-smollm2.json
```

On failure, the CLI returns a nonzero exit and an `UNRESOLVED` report, rather than silently substituting fixture calculations. No dependency is added to the 65,536-byte compact core.

## Benchmarks for the next frontier

Evaluate a pinned, instruction-tuned model with actual role-aware chat tokenization, multi-turn continuation, and independent prompt/answer quality scores. Compare tokens per second, host calls, transferred activation bytes, and cost to resident-host inference. Add per-layer operator adapters for Qwen Gated DeltaNet, DeepSeek's MoE/sparse attention and smaller Llama decoders under the procedural graph, then measure actual multi-host operator execution and process recovery. Expand bounded agent tasks with explicit approval/evaluation contracts. The main question is whether operational Grounds reduce redundant interpretation and improve model navigability, not whether a checksum compresses arbitrary trained mathematics.
