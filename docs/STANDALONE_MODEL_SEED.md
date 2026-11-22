# Running a trained decoder from the VL-64 model seed

The model profile packages the VL-64 operations needed to read a Safetensors header, construct a typed Llama execution graph, acquire bounded weight ranges and evaluate token IDs. The same executable accepts a local directory or a public Hugging Face repository at an immutable revision. A host with Python 3.11 or later can run the archive using its standard library.

The measured build occupies **24,109 bytes**. Counting every expanded Python source file in the archive, including its verifier, gives **58,740 bytes**. The builder enforces a 65,536-byte limit on each measure. The checkpoint, working memory and host Python installation have separate costs. The general-purpose VL-64 seed and the model profile use the same source definitions for their shared operators.

## Try the executable

The distributed archive was built with CPython 3.14.0 and reproduced byte-for-byte with that builder. A CPython 3.13.3 build produced identical expanded archive members in a 24,081-byte ZIP. The original checkpoint comparison used Python 3.13.3 to run the distributed archive.

```sh
python -m experiments.seed_fabric.build_model --out artifacts/VL64-model.pyz
python -I -S artifacts/VL64-model.pyz seed-verify
python -I -S artifacts/VL64-model.pyz inspect --hub HuggingFaceTB/SmolLM-1.7B --revision d7449ff7241c863f3e8accc475155f0f97afa011
python -I -S artifacts/VL64-model.pyz run --snapshot "/path/to/SmolLM-135M" --tokens 42 75 --steps 2
python -I -S artifacts/VL64-model.pyz run --snapshot "/path/to/SmolLM-135M" --tokens 42 75 --steps 2 --pair 9805 787
```

Use `--include-logits` for a full vocabulary row per generated position. The user supplies token IDs and a step count. Each step evaluates the supplied and generated prefix afresh and selects the highest-scoring token, using the lowest ID for a full-vocabulary tie. The restricted pair mode stops on a zero score. For a numerical comparison against Transformers, run the optional [reference harness](../experiments/model_fabric/model_seed_reference.py) with `--seed`, `--snapshot` and `--out`.

## What the glance determines

Inspection binds the config, index when present, tensor shapes, data types and header hashes. For sharded files it checks index ownership against the actual headers and maps every virtual tensor interval back to one physical file. A live inspection of SmolLM-1.7B read **43,637 metadata bytes**, observed **218 tensors** across two shards and mapped **6,845,505,536 payload bytes** with **zero weight bytes read**. Execution acquires the actual values through requests of at most 65,536 bytes. Local range receipts identify observed bytes. Reproducing a particular checkpoint also requires a trusted source pin.

The implemented numerical contract covers the admitted bias-free Llama graph with ordinary RoPE, grouped-query attention, RMSNorm, SwiGLU and F32 or BF16 weights. Dimensions come from the snapshot within the decoder's explicit bounds. Inspection can report an unsupported architecture before inference. The 1.7B measurement covers topology only. [Hosted checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B/tree/d7449ff7241c863f3e8accc475155f0f97afa011)

## Measured generation and restricted readout

Two isolated seed processes ran from a temporary directory with site packages disabled. The reference process separately loaded the pinned SmolLM-135M weights in Transformers and evaluated them in CPU FP32. The seed used Python binary64 arithmetic.

| Observer after input IDs 42, 75 | Two generated IDs | Independent comparison |
| --- | --- | --- |
| Full vocabulary | 32, 28 | All 49,152 logits checked at each position. Maximum absolute errors were 0.000021864 and 0.000017465. |
| Ordered pair “ yes” and “ no” | 787, 787 | The FP32 reference preferred “ no” on both restricted prefixes. |

For the pair, let the final residual be $h=u+Dv$. Given output rows $w_i,w_j$ and final RMSNorm gain $g$, compile $a=(w_i-w_j)\odot g$ and $b=D^\top a$. The sign of $a^\top u+b^\top v$ gives the preference in real arithmetic because the shared RMS divisor is positive. Compiling the directions read 3,545,856 bytes. Each terminal query could then stop before the last down-projection and vocabulary head. Finite-precision implementations can disagree near a tie.

Generating the two full-vocabulary IDs took **264.7 seconds** and acquired **2,350,566,144 logical weight bytes** including repeated prefix work. The restricted run took 224.4 seconds. The runs were concurrent and answered different questions, so these timings do not establish a speedup. Persisting the causal key/value state between generated positions would directly address the repeated work. [Source identities, per-step costs and full comparison](../experiments/model_fabric/datasets/2026-11-22-isolated-smollm-135m-two-step-reference.json)
