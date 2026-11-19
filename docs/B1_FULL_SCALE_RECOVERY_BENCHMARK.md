# VL-64 full-scale B1 benchmark: external model hosts, genuine interpreter recovery

**Experimental scope:** This is a **B1 / VL-mediated** test under [VL-I's experiment protocol](https://github.com/Kier73/VL-I/blob/main/docs/EXPERIMENT_PROTOCOL.md). It is not B2 (VL-native weight-backed numerical realization) or B3 (certified predictive shortcut). The separate [VL-I-001](https://github.com/Kier73/VL-I/issues/1) effort pursues those additional grades. Neither grade can be inferred from this one.

The two unchanged model serving **routes** are `Qwen/Qwen3.8-27B` on Hugging Face's router and `deepseek-flash` on DeepSeek's API, claiming `DeepSeek-V4.1-Flash-1017`. They are **not** verified immutable model snapshots; the provider may serve changing underlying weights. VL-64 does not silently replace either with a smaller model. The GitHub CPU runner never downloads neural weights or calls PyTorch to run the model. Chat-completions are provider-side work; provider logit/KV state cannot be inspected, proven identical or recovered from this interface.

## Benchmark sequence, both models

Each **model route** runs in two distinct Python interpreter processes. Both routes run concurrently, keeping their provider Ground identities and receipts isolated. A new process begins from the externally protected checkpoint digest—not merely from a self-asserted saved state.

| Epoch | Procedure | Evidence |
|---|---|---|
| 0 | Construct admissible model-route Ground; request coherent paragraph; validate shape and keyword heuristic; commit transcript; persist exact checkpoint | Provider-returned text, optional token usage, input/output frame sizes, process CPU time, remote wall time, checkpoint digest |
| 1 | Start a **new interpreter process**, verify previous checkpoint and its entire receipt chain; request safe Python function; require the *correct* zero-denominator branch structurally without executing model code; request exactly one authorized `add_numbers(3,4)` proposal; compute `7` in the VL caller; ask for an explicit final `7`; commit new receipts | Verified previous epoch and no replay, restricted code AST, authorized action result, final completion, fresh checkpoint and CPU timing |

If the model returns an invalid output, code, or action, record unresolved/false instead of inflating success. The small suite measures paragraph topical coverage only via a heuristic (it does not prove factual correctness). Python's task-specific static evaluation checks the control structure against the intended semantics, not arbitrary program execution. Model-generated Python is never executed. The agent has only a pure arithmetic action; it has **no** shell, network or filesystem permission.

## Reported work and limitations

`experiments.model_fabric.remote_large_recovery_benchmark` reports the two model routes, all four process epochs, receipts, model identities **as claimed by the provider**, independent checkpoint digests, per-task completion usage if supplied, CPU time spent **inside the VL process**, provider request wall time, traffic bytes, quality indicators, and total fleet wall time.

A **reported generation rate** is `sum(completion_tokens) / sum(provider-request wall seconds)` for each route; it includes provider queuing and networking, not just accelerator time. If any token count is unavailable, this rate is `null`. This cannot be used as a model-internal FLOPS benchmark. Provider inference time and CPU interpreter time are explicitly separate. The result is not a universal model benchmark: one prompt per task, no statistical uncertainty, and a very small arithmetic-only agent action.

**Recovery** here is continuity of VL-64's control transcript and authority, with a new controller process; the provider's internal KV or model process is not restored. There is no duplicate invocation of the paragraph request during recovery. To prove host-state portability, a future operator-level provider must expose recoverable causal state and independent numerical challenges.

## Reproduce

Fixtures (no trained model inference, offline, no provider credentials):

```bash
python -m pytest -q tests/test_remote_large_fleet.py tests/test_remote_large_recovery_benchmark.py
python -m experiments.model_fabric.remote_large_recovery_benchmark \
  --fixture --max-tokens 128 --working-dir artifacts/b1-fixture-work \
  --out artifacts/b1-fixture.json
```

Live provider calls (opt-in and may incur costs, original full-scale routes only):

```bash
export HF_TOKEN='...'
export DEEPSEEK_API_KEY='...'
python -m experiments.model_fabric.remote_large_recovery_benchmark \
  --live --max-tokens 128 --working-dir artifacts/b1-live-work \
  --out artifacts/b1-live.json
```

To use GitHub Actions, dispatch **Full-scale VL64 B1 recovery benchmark** with `run_live=true` once repository secrets are configured. The feature PR is also permitted to attempt a single credentials-gated evaluation, with no model fallback and no credentials printed. Missing credentials generate an explicit unresolved record, not a success receipt. GitHub Actions fixture tests run on Python 3.11 and 3.13.

## What VL-I can take from this

The exact B1 report schema, adversarial negative controls, cost measurement dimensions, action authority and checkpoint verifier can serve as a **comparison baseline**, not an implementation of VL-I's intended native model realization. A future B2/B3 run should use the **same model and same intention**, with pinned weight/tokenizer identity and independent output parity, to establish real saved work. A decrease in VL process CPU time alone does not demonstrate savings in the host's trained mathematics.
