# Intention-oriented virtual inference and model fleets

The Virtual Layer is a **control and interpretation substrate**; it need not own the trained weights. It controls how observer-hosts' tokenizers and model operators are addressed, composes the intended operation, validates bounded result frames and preserves causal lineage. The trained mathematics still executes at a capable host.

## The banks of the river

`compile_intention` binds a qualified operational model Ground to its tokenizer identity, role messages, output kind, generation and byte budgets, explicit stop-token policy, and artifact identity. A canonical digest covers the complete request. An `IntentionSession` then:

1. Renders role messages into an escaped *plain-role* prompt or a native-chat tokenizer envelope.
2. Uses the admitted **model-host tokenizer** (no universal token vocabulary or hard-coded chat tokens).
3. Opens the model's real causal prefix; receives all logits in frames of at most 4,096 F32 values (16 KiB of payload), checks range integrity and the complete-vocabulary SHA-256, and asks the host to advance using the independently selected token.
4. Applies an *output-shape contract*: paragraph form, parsable Python **without executing it**, or strict JSON tool/final decisions. This is **not** a semantic, usefulness, safety, or training-quality certificate.
5. Saves the intent and model checkpoint digests. No learned weight bytes enter these records.

Native HF chat templating is opt-in through `HostedNativeChatTransformer` and the `hf_native_chat_template_v1` tokenizer identity. The model host must implement `apply_chat_template`. The Virtual Layer refuses substitution of plain text or fabricated special token IDs in native-chat mode. Ordinary text-role formatting is a portability fallback and is *not equivalent* to a model's training-time chat format.

## Multiple VL model instances and agent actions

`ModelInstanceFleet` admits independent model hosts, pins each Ground separately, and isolates token prefixes, vocabulary, tokenizer identity and checkpoints. It supports bounded parallel requests. The tests launch two distinct Python processes as well as parallel, differently tasked model hosts. This is **not** proof of fully federated network-host execution or protection against malicious host claims. Existing E-CPU mutual-TLS and quorum-fencing contracts remain candidates for future transport-level integration.

`AuthorizedIntentAgent` runs bounded JSON decisions and invokes **only caller-supplied tool functions**. Unknown tool requests fail; output byte, argument, and round budgets are enforced. Tools are never implicitly discovered from the model or executed just because the model emits their names. The fixture benchmark demonstrates a useful arithmetic action, not unrestricted autonomy.

## Evidence quality and benchmark tiers

- **Protocol fixture**: trained-output-shaped scripted hosts verify correctness, role escaping, code-syntax checks, action refusal/approval, multi-model isolation, and process separation. Scripted outputs are **not neural language quality benchmarks**.
- **Real trained byte language model**: existing `HostedTrainedByteLanguage` retains its genuine count-trained transition weights; an independent reference checks emitted token identities. Its output may fail the paragraph contract, which is recorded as `OUTPUT_CONTRACT_UNRESOLVED` rather than dressed up as conversational intelligence.
- **Real pretrained instruction model**: opt-in through `HostedNativeChatTransformer` with an immutable 40-character revision and exact native chat tokenizer. Requires real host runtime / model bytes; record instruction-following, code correctness, task success, token throughput and checkpoint recovery independently. Passing protocol tests does not satisfy this tier.

## 64 KiB core debug pass

The core audit tightens canonical JSON hashing to reject non-finite floats (NaN / infinities). Fixed-size copy-on-write `PageSpace` now rejects negative page coordinates and writes that would grow or silently shrink a 4,096-byte page. Regression tests cover both. The `src/vld` source byte budget remains at or below **65,536 bytes**, checked in CI; core changes are measured from repository source bytes, not just the zip package.

## Commands

```bash
python -m pytest -q tests/test_intention_fabric.py tests/test_intention_trained_byte.py tests/test_core_integrity_bounds.py tests/test_host_resident_kv.py tests/test_operational_ground.py tests/test_operational_tensor_graph.py
python -m experiments.model_fabric.intention_fleet_lab --out artifacts/intent-fleet.json
```

The development should next connect output-shape tests to independent answer/coding evaluations on a capable, instruction-tuned model, and authenticated host transfer. The Virtual Layer's operational Ground can guide the **intention**, but it cannot create model capabilities absent from the weights or turn untrusted host descriptions into proof.

## Host-resident KV reuse

The optional `HostedTransformersLanguage` now requests `use_cache=True` from the model runtime. When and only when a new token prefix equals the previous committed prefix plus one token, it supplies the host-owned `past_key_values` and computes the new token from a one-token input. A different prefix, missing cache, or failed operation forces a full prefill and never publishes a stale causal cache head. This improves the mathematical work for successive tokens, **not the byte size of the VL core**. Synthetic fake-model tests independently check `[full prefill, 1, 1]` input lengths, sibling-prefix invalidation, model failures and runtimes that do not support KV. The actual speedup depends on architecture, runtime and host hardware and is not inferred from those fixtures.

### Optional pinned instruction-model observation

The feature-branch PR workflow attempts three real-model tasks with `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`: a paragraph, Python code, and one approved `add_numbers` agent action. Each writes an independent artifact and is **non-blocking**; failures are reported as unresolved, not fabricated outputs. The host records actual successful causal KV prefill/reuse counts. An instruction-model shape pass does not establish semantic quality, code correctness, or broad autonomy.
