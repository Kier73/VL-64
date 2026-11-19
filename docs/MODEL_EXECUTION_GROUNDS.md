# Language Models, Pretrained Tensor Navigation and the VL-64 Model Seed

**Status:** Executed experiments, independently replayed weight-bearing operations and reproducible model seed.  
**Core:** The seven `src/vld/*.py` modules remain exactly **65,093 / 65,536 source bytes**. All new operators, adapters, acquisition plans and model experiments live in [model_fabric](../experiments/model_fabric/) outside the compact runtime.

## Goal

An external observer should be able to describe a desired model task to VL-64, rather than telling it which server, processor or code path to use. The Virtual Layer should:

1. **Read** an arbitrary model's actual byte representation and extract typed structure, without assuming a particular neural architecture.
2. **Build a computational Ground** that records tensor identity, dimensions, data types, locations, operator requirements and evidence.
3. **Select a qualified realization**, such as a verified local operator, an admitted remote action, or an independently verifiable slice of an external weight file.
4. **Compute the requested result** and return the selected format with a receipt distinguishing mathematical prediction from actual weight-backed execution.

This iteration demonstrates all four steps on specific, measured examples. It does not yet execute an entire arbitrary pretrained Hugging Face model from header information alone. An exact forward pass requires the actual values of every weight and activation on its causal path, or an independently admitted remote evaluator.

## Experiment A — a trained byte-level language model

The first implementation is a real **first-order byte language model**, trained directly from 80% of the committed VL-64 README bytes. Each pair of adjacent training bytes becomes an exact count-valued transition weight `W[next_byte, previous_byte]`. The learned sparse weight matrix is stored in a content-addressed Ground.

It is evaluated using the existing **`vld.StructuralLinearOperator.matvec`** utility in the compact runtime, with a one-hot byte input and add-one probability smoothing. Generation repeatedly chooses the highest-probability next byte. There is no neural-network training claim, no tokenizer other than identity bytes, and no simulated logits.

Measured at the recorded source revision:

| Measurement | Result |
| --- | ---: |
| Training bytes | **13,597** |
| Independent held-out bytes | **3,400** |
| Sparse Ground JSON description | **13,115 bytes** |
| Held-out cross entropy | **4.447546 bits / byte** |
| Held-out perplexity | **21.819496** |
| Uniform-byte baseline | 8 bits / byte |
| Held-out top-one next-byte accuracy | **25.4487%** |
| Actual generated continuation | Repetitive `erererer...` |

The repetitive output is reported intentionally. The statistical language model genuinely predicts bytes and is better calibrated than a uniform baseline on this held-out sample, but is far less expressive than a transformer or a conversational model.

An independent implementation using a plain count dictionary checks the first predicted byte, while separate tests verify positive probability normalization, generated sequence length, content fingerprints and tamper rejection.

The external observer can submit the pre-generated Ground to the existing `ObserverIntention` interface:

~~~bash
python -m experiments.model_fabric.model_lab --out artifacts
python -m experiments.observer_intent.arbitrate \
  --intent artifacts/language-intention.json \
  --out artifacts/language-generation.json
~~~

The operation is `model.byte.generate`, with a hex-encoded prompt and exact requested number of output bytes.

## Experiment B — real F32 causal-attention execution

A second experiment constructs a small **single-head causal-attention decoder** from exact F32 tensors stored in a real Safetensors file (962 total bytes, including metadata). This is a small numerical model with a vocabulary of eight tokens, hidden width four, and real Q/K/V and output projections.

Its admitted operator path is:

~~~text
token embedding
    ↓
RMSNorm
    ↓
Q, K and V structural linear operators
    ↓
causal scaled-dot-product softmax attention
    ↓
output projection and residual addition
    ↓
RMSNorm and vocabulary projection
    ↓
argmax next-token choice
~~~

The Q/K/V, output and vocabulary matrix-vector products run through VL-64's compact `StructuralLinearOperator`; the normalizations use the existing compact `rms_norm` implementation.

Tests separately reconstruct the same actual F32 tensor values and perform **independent scalar matrix, softmax and attention calculations** without calling VL-64's operator methods. Outputs agree within an explicit numerical tolerance over several input sequences.

The model generates token IDs, but its weights are *seeded demonstration weights*, not pretrained text weights. This establishes genuine **weight-bearing neural operator execution** through the Virtual Layer, not useful text fluency.

The observer operation `model.neural.generate` reads the locally admitted Safetensors payload and returns a typed result. Incomplete weight metadata is insufficient for execution and produces an unresolved result.

## Experiment C — inspect an actual pretrained model without weight download

The test target was the public, pinned [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) checkpoint, revision:

`4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`

The model is distributed under the Apache 2.0 license. This is a reproducible benchmark, **not a claim that it is the newest model on Hugging Face**.

The `hf_glance.py` adapter used HTTPS to acquire the small `config.json` and two bounded Range requests for the Safetensors header. The existing compact `SafetensorsGlance` interpreted the eight-byte little-endian length and the JSON tensor manifest; the external adapter checked duplicate keys, dtype/element sizes, byte offsets, non-overlap and announced file extent.

**Actual observed structure:**

| Measurement | Observed |
| --- | ---: |
| Config bytes | **704** |
| Safetensors header bytes | **30,536** |
| Tensor entries | **272** |
| Declared BF16 weight payload | **269,030,016 bytes** |
| Actual weight payload downloaded during structure scan | **0 bytes** |
| Model family | `LlamaForCausalLM` |
| Hidden width | 576 |
| Transformer layers | 30 |
| Attention heads / key-value heads | 9 / 3 |
| Vocabulary | 49,152 |

This Ground is a **verified description of tensor shape and byte location**. It does not contain trained tensor values and therefore cannot by itself produce genuine SmolLM2 logits.

The reader deliberately refuses to treat an HTTP 200 full-file response as a valid partial range response. If the provider will not honor bounded byte ranges, that observation is unresolved rather than silently downloading the complete checkpoint.

The stored [pinned tensor structure Ground](../experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json) is 46,132 bytes including its newline, despite describing the byte locations and dimensions of 272 tensors in a roughly 269 MB payload.

## Experiment D — execute a genuine pretrained operator from the Internet

Using the actual observed tensor offsets, VL-64 selected **only two required weight regions** from the first SmolLM2 layer:

| Selected payload | Exact bytes read |
| --- | ---: |
| `model.layers.0.input_layernorm.weight` — 576 BF16 values | 1,152 |
| First four rows of `model.layers.0.self_attn.q_proj.weight` — 4 × 576 BF16 values | 4,608 |
| **Total actual pretrained weight bytes** | **5,760** |

After acquiring those exact source ranges, the Virtual Layer decoded BF16 values and executed:

\[
y_{0:4}=W^{(0)}_{Q,\ 0:4}\ \mathrm{RMSNorm}(x,\gamma^{(0)},\varepsilon)
\]

with a deterministic synthetic 576-dimensional input vector `x`. This expression is an **actual weight-bearing sub-operation** of the pinned pretrained model. It is evaluated by VL-64's `rms_norm` and `StructuralLinearOperator.matvec`, then compared against an independent scalar reference.

The four observed output coordinates were:

\[
[-0.198187536,\,-0.288741521,\,-0.397372802,\,0.612992252]
\]

Both Python 3.11 and 3.13 produced identical nine-decimal results and independently verified the same source byte fingerprints. The chosen slices comprise approximately **0.002141%** of the full model's declared weight payload. The header and configuration were read separately to locate those slices.

**Important:** this does *not* produce SmolLM2 text, logits or a real token's first-layer Q vector. The input vector is explicitly synthetic, and only four projection rows were evaluated. RoPE, remaining Q/K/V vectors, attention, all feed-forward layers, other norms, tokenization and the final vocabulary projection were not executed. It establishes direct communication-mediated realization of an authentic pretrained tensor subgraph.

## A portable Ground of actual pretrained weights

The two observed BF16 byte slices are preserved together with their hashes, exact byte intervals, model revision, declared dimensions and typed operator relation in a **9,229-byte content-addressed Ground**:

[Download the source-traceable Ground description from the repository](../experiments/model_fabric/datasets/2026-11-18-pretrained-bf16-slice-ground.json).

Another host can independently load the Ground and call `replay_slice_ground`. Its scalar output is checked against the same independent reference *without contacting Hugging Face*. This is a concrete example of the desired separation between **structural interpretation**, **needed payload bytes**, and **where/when realization happens**.

A hash proves consistency with the supplied Ground bytes, not the authenticity of a remote publisher. The recorded pinned revision and HTTPS Range provenance are independent supporting observations.

## The monolithic model seed

The strongest result packages the original **65,093-byte VL-64 runtime source**, its generated software Ground of Grounds, this exact pretrained weight slice, and an audited replay entrypoint into one deterministic Python zipapp.

| Measurement | Verified |
| --- | ---: |
| Standalone pretrained model seed | **38,360 bytes** |
| Source runtime included in compressed form | 65,093 original bytes |
| Exact pretrained tensor payload included | 5,760 bytes |
| Seed bytes remaining to the 65,536 ceiling | **27,176 bytes** |
| Compression of source to seed | The source, carrier, code and Ground all fit together |
| Independent tested hosts | Linux / Windows |
| Python versions | 3.11 / 3.13 |
| Test result | **249 passed per environment** |
| Seed ZIP SHA-256 | `1fa19379ff8f4df46acc019ae086d46277c29ac849b003885c5c718ad49240d8` |

On every host the seed ran with Python's isolated-mode `-I` flag and **without repository checkout, additional package installation or network access**, reproducing exactly the same four numeric outputs and receipt fingerprint.

Build and run:

~~~bash
python -m experiments.model_fabric.model_seed \
  --out artifacts/VL64-ModelSeed.pyz

python -I artifacts/VL64-ModelSeed.pyz model-verify
python -I artifacts/VL64-ModelSeed.pyz model-run
python -I artifacts/VL64-ModelSeed.pyz model-ground

# The existing VL-64 utilities also remain available:
python -I artifacts/VL64-ModelSeed.pyz verify
~~~

This model seed is *not* a Python-free virtual machine or a full SmolLM2 checkpoint. It is the first executable, portable **piece of a real pretrained language-model computation** whose source, structural description and necessary tensor values all fit inside the 64 KiB envelope.

The [monolithic pretrained-model seed CI workflow](../.github/workflows/model-seed.yml) tests Linux/Windows × Python 3.11/3.13, byte equality of the archive, exact source reconstruction, self-Ground verification, model execution and full earlier regression suites. Its artifacts include the actual executable `VL64-ModelSeed.pyz`.

## Next work to reach full model inference

1. **Typed execution DAG from metadata:** identify the exact architectural semantics of each operator, distinguish known patterns from merely similar tensor names, and preserve all unresolved obligations. A generic parser must not treat every neural architecture as Llama.
2. **Byte-range tensor planning:** construct the minimal set of needed tensor ranges for an observer's requested intermediate or final output. Some final outputs depend on essentially every pretrained weight and cannot be recovered exactly from a tiny header.
3. **True tokenization and embeddings:** capture tokenizers, embedding lookups, positional encodings, vocabulary projection, numerical conventions and input/output text formats. These are necessary for genuine model-generated text.
4. **Full decoder realization:** implement and verify RoPE, grouped-query attention, SwiGLU/feed-forward operators, KV state and residual connections, with independent reference comparisons at each layer.
5. **Federated remote model hosts:** use an authorized distributed network of model-capable observers to evaluate selected operators, returning typed receipts and costs. Moving the work to a remote host still uses that host's actual compute and memory.
6. **Compression and navigation research:** look for verified low-rank, quantized, structured or observationally equivalent Grounds that can replace full tensor payloads for particular intentions. Measure actual predictive equivalence and full costs; do not assume tensor shapes alone reconstruct the values.
7. **Model-seed synthesis:** project the software Ground and only the necessary tensor fragments into smaller specialized seeds, optimizing byte transport, host capability, output fidelity, numerical cost and evidence.

The measurement that matters is not how much of a model we can *describe* without its weights; it is how much of a requested **verified computation** we can faithfully realize with only the information and host capabilities the particular intention actually requires.

## Sources and research heritage

- [Safetensors format](https://github.com/huggingface/safetensors#format) — length-prefixed tensor metadata, data offsets and payload separation.
- [Hugging Face: metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing) — bounded Range retrieval for tensor structure.
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — pretrained model source and license.
- [VL-64 compact operators](../src/vld/micro_operators.py) — actual matrix-vector and RMS normalization primitives.
- [In-Formation-Space neural execution research](https://github.com/Kier73/In-Formation-Space/tree/main/bridges/neural-observation-ground) — prior typed neural operator DAG experiments and verification methods.

All numerical values above are backed by the versioned [model execution evidence](../experiments/model_fabric/datasets/2026-11-18-model-execution-evidence.json), [trained Ground](../experiments/model_fabric/datasets/2026-11-18-trained-byte-language-ground.json), [remote tensor Ground](../experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json), [pretrained slice Ground](../experiments/model_fabric/datasets/2026-11-18-pretrained-bf16-slice-ground.json), and CI runs in this repository.
