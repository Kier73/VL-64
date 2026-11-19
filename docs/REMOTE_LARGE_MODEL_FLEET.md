# VL-64: CPU-only interpretation for the original large instruction models

The benchmark deliberately keeps **Qwen/Qwen3.8-27B** and **deepseek-ai/DeepSeek-V4.1-Flash-1017**. It does not download, quantize, emulate, replace, or execute either large neural model on the GitHub CPU runner. Each model's trained mathematics and tokenizer stay at an external model provider.

The compact `src/vld/` runtime is unchanged. This experiment lives in `experiments/model_fabric/remote_large_fleet.py` and composes the existing `intention_fabric` output contracts with `operational_ground` canonical evidence hashes.

## Two distinct execution grades

| Grade | Interface | What VL-64 observes and operates | What is *not* established |
|---|---|---|---|
| **Remote chat provider** (this work) | HTTPS chat completions | Intent → role messages → model-route Ground → response validation → caller-approved arithmetic action → checkpointed receipt; separate CPU vs remote wall timings | Raw tokenization, per-position logits, exact pinned revision, certified host computation, KV-transfer continuity |
| **Native operational host** (pre-existing) | `OperationalModelNavigator` `encode/open/logits_tile/advance` | Qualified tokenizer IDs, causal token positions, complete F32 logit tiles and hash-checked greedy token choices | That an arbitrary third-party hosted chat endpoint implements the native operator RPC or that its self-report proves trained arithmetic |

Do **not** label chat-provider output as a full token-level VL64 forward pass. The host is a real trained model service but the model's internals are opaque to this route. Both grades share the *intention*, explicit model identity, authorized-action policy, and evidence disciplines. An external operator-capable E-CPU/Virtual-Layer host can later supply the stronger RPC without moving model weights to the VL.

## Model routes (no smaller substitutes)

- **Qwen3.8-27B** → official Hugging Face Inference Providers router, model name `Qwen/Qwen3.8-27B`, `HF_TOKEN` credential. Endpoint: `https://router.huggingface.co/v1/chat/completions`.
- **DeepSeek-V4.1-Flash-1017** → official DeepSeek API, route `deepseek-flash`, `DEEPSEEK_API_KEY` credential. Endpoint: `https://api.deepseek.com/chat/completions`.

These are **provider serving aliases**. They are not independent SHA-256 attestations of the exact underlying checkpoint revision. The provider-reported served name is retained as an observation. Requests use strict HTTPS routes, do not follow HTTP redirects, and never store bearer tokens in Grounds or receipts. The CPU runner has no model-specific tensor dependencies.

## Benchmark suite

Both models receive the same three intentions:

1. **Paragraph:** explain why observing a trained-weight description does not itself execute inference and what a receipt proves.
2. **Code:** produce a pure `safe_divide(a,b)` implementation with zero-denominator handling. Syntax, restricted AST shape and task-specific structure are tested; model-generated code is **not executed**.
3. **Agent:** request `add_numbers(a=3,b=4)` through a strictly validated JSON proposal. VL-64 exposes only this pure arithmetic tool and requires the model to finalize with `{"final":"7"}`. No shell, network, or filesystem permissions are given to the model.

Each receipt carries ground and intent digests, actual provider response digest, provider-reported model, response-shape status, a distinct task heuristic, provider usage token counts when supplied, remote wall time, interpreter **process CPU time**, request and response byte counts, and continuity in a hash-chained transcript. Response frames never exceed 64 KiB; request frames are capped at 8 KiB. It does not count model weights transferred to VL (always zero).

A checkpoint includes **only the VL transcript**. A fresh interpreter process can restore its exact externally pinned digest and continue with its own new request. The opaque provider's KV state **cannot** be transferred or verified, and this checkpoint is not a proof that a provider will reproduce the exact same tokens under retry. The test does not silently reissue a completed request.

## Repeatable commands

Offline protocol-only test on any CPU runner, without provider secrets or pretrained computation:

```bash
python -m pytest -q tests/test_remote_large_fleet.py
python -m experiments.model_fabric.remote_large_fleet_lab --fixture --out artifacts/remote-large-fleet-fixture.json
# One separate VL interpreter process for each model:
python -m experiments.model_fabric.remote_large_fleet_lab --fixture --instance qwen-27b --out artifacts/qwen-control.json
python -m experiments.model_fabric.remote_large_fleet_lab --fixture --instance deepseek-v41-flash --out artifacts/deepseek-control.json
```

Actual provider model inference (opt in explicitly; model API costs may apply):

```bash
export HF_TOKEN='...'
export DEEPSEEK_API_KEY='...'
python -m experiments.model_fabric.remote_large_fleet_lab --live --out artifacts/remote-large-fleet-live.json
```

The GitHub Actions workflow runs **only fixtures** on ordinary pushes/PRs. The provider job runs **only** on manually dispatched workflow with `run_live=true` and configured repository secrets `HF_TOKEN` and `DEEPSEEK_API_KEY`. The workflow must fail closed if either secret is unavailable; **absence of an authenticated provider report is not inference success**.

A live run can measure application-level provider throughput where providers return token usage and host API wall time. It cannot isolate GPU computation from transport/queueing inside a third-party service; reported `vl_cpu_seconds` measures only the runner process CPU consumption. The hypothesis that VL interpretation is cheaper than executing 27B/DeepSeek-scale mathematics is sound directionally but needs measured overhead, throughput, and host attestations before any hard performance claim.

## What follows

Use E-CPU authenticated communications to connect remote operator-capable hosts implementing `encode/open/logits_tile/advance` (or parameter-specific tensor operations) rather than remaining limited to chat APIs. That is the route toward the intended **Virtual Layer as a true procedural processor**, while large model weights never reside in its compact CPU interpreter.

References: [Hugging Face Inference Providers API](https://huggingface.co/docs/inference-providers/index); [DeepSeek first API call](https://api-docs.deepseek.com/en/); [existing VL-64 operational model Grounds](OPERATIONAL_MODEL_GROUNDS.md).
