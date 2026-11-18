# Observer Intention → Qualified Ground → Result

**Status:** Initial implementation and research direction, November 18, 2026.  
**Implementation:** `vld.ObserverIntention`, plus the [external observer-intention lab](../experiments/observer_intent/README.md).  
**Core source budget:** 65,536 bytes for the VL-64 package. Acquisition and integration remain external.

## Purpose

The **external observer** chooses an intention and a desired result. VL-64 is the communicable middleman between that request and qualified state-space descriptions/operations. The external observer should not need to name the DNS query, Git ref request, model metadata file, or HTTP endpoint sequence required to realize the intention.

This does not require natural-language understanding. A small typed request is an explicit, versionable language of intentions. Natural language may be translated into it later by an optional untrusted front end; the receiving runtime must check the typed contract and independently qualify the selected operation.

**Current demonstrated path:**

~~~text
observer JSON request
    ↓ type and bound check, intention digest
VL-64 ObserverIntention
    ↓ known read-only operator + plan digest
external resource adapter
    ↓ Hugging Face metadata / pinned config or GitHub observed relations
content-addressed Ground + observation evidence
    ↓ format requested by observer
JSON / JSONL / text result
~~~

The compact core **does not automatically execute URLs or external code**. Its role is to represent an intention without tying it to one interpreter, host, or substrate. The lab supplies registered network operators.

## Example intentions

### Observe the latest *modified* public text-generation entry

~~~json
{
  "operation": "hf.model.describe",
  "target": "latest:text-generation",
  "output": "json",
  "options": {
    "metadata_only": true,
    "catalog_limit": 12
  }
}
~~~

This queries a bounded catalogue ordered by `lastModified`, chooses the first public eligible entry and returns its metadata Ground. It does **not** mean the newest foundational model, latest newly published architecture, best model, or most recently trained weights. The returned model identity, revision and captured time identify precisely what was observed.

An entry consisting of checkpoint artifacts with no accessible architecture config is correctly a `PARTIAL_MODEL_STRUCTURE`. In a November 18 live run, a recently modified training-state repository fell into that category.

### Require a usable architecture, not merely a recent listing

~~~json
{
  "operation": "hf.model.describe",
  "target": "latest:text-generation",
  "output": "json",
  "options": {
    "metadata_only": true,
    "catalog_limit": 12,
    "require_architecture": true,
    "max_candidates": 6
  }
}
~~~

The arbitrator inspects the top bounded candidates in order. It selects the first candidate with successfully observed architecture fields and reports the selected rank, all attempted candidates and their refusal/partial reasons. If none qualifies, it returns `UNRESOLVED` rather than lowering the requested criterion without permission.

The experimental request can also specify `library=transformers`, narrowing the catalogue to candidates that advertise that library. A successful November 18 run selected the fourth candidate that exposed configuration fields. That remains an observed partial architectural description, not a qualified inference realizer.

This is a useful property of an intention-aware system: **different observer requirements result in different resource navigation**, even when both requests begin with the same external catalogue.

### Inspect a specific model in a selected output format

~~~json
{
  "operation": "hf.model.describe",
  "target": "openai-community/gpt2",
  "output": "text",
  "options": {"metadata_only": true}
}
~~~

The live reference result is grounded to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Its declared configuration includes `model_type=gpt2`, `n_embd=768`, `n_layer=12`, `n_head=12` and `vocab_size=50257`. These are configuration facts, not proof that trained weights were acquired or executed.

The same request can select `json` or `jsonl`; the requested representation is not silently chosen by the adapter.

## Protocol contracts

### Request

\[
I=(o,\tau,F,\kappa)
\]

- `o`: registered operation (e.g. `hf.model.describe`);
- `\tau`: target or explicit selector (e.g. repository identifier or bounded recent-catalogue query);
- `F`: required output format;
- `\kappa`: options and acceptance constraints (metadata-only, bounded search, architecture required).

The request is validated and content-addressed as `vl64.observer-intention.v1`. Unknown operators are unresolved. Options are bounded and cannot themselves act as code. An intention digest binds the exact request, but **does not authorize a remote action**.

### Plan

The adapter constructs `ADMITTED_READ_ONLY_PLAN` plus a digest-bound sequence such as:

~~~text
query_catalogue_latest_modified
 → read_model_info
 → read_pinned_config_if_listed
 → form_metadata_ground
 → format_observation_receipt
~~~

The current planner is deterministic and operation-specific, not yet a general multi-provider optimizer. The term `ADMITTED_READ_ONLY_PLAN` describes this local fixed policy, not a proof of a model's validity or remote service authority.

### Ground and observation receipt

The Hugging Face model Ground contains:
- model ID and pinned repository commit;
- listing/catalogue interpretation and selected rank;
- declared architecture/config fields when observed;
- weight *filenames* and available Safetensors summary metadata;
- nodes and relationships linking model, revision, config and file manifest;
- content digests of the raw metadata response and explicitly scoped source URLs;
- the operation's unobserved aspects, including no model inference and no weight payload.

The external result includes observed response metadata and the selected output format. Responses are structured as `OBSERVED_MODEL_STRUCTURE`, `PARTIAL_MODEL_STRUCTURE`, `UNRESOLVED`, or `REFUSED`. These statuses are not interchangeable.

### Observer-selected return

For a model-inspection intention, the observer can ask for a human summary, structured machine-readable receipt or JSONL event representation. In all cases the output should include the same core identity, evidence and confidence claims. The byte encoding and digest of the chosen output should also be reported.

## The model-execution distinction

A language model is not generally specified by architecture and tensor shapes alone. Its trained outputs depend on **the actual weight values**, tokenization, runtime semantics, sampling method and other model-specific data.

Thus:

| Requested behavior | Metadata-only capability | Additional realization needed |
| --- | --- | --- |
| Discover model identity and revision | Available | None |
| Inspect declared configuration and filenames | Available when present | None |
| Describe possible operator topology | Partially inferable from model-family contracts | Explicit family adapter and validation |
| Recover exact trained tensor values | Not established | Read authorized weights or a separately proven reconstructive representation |
| Execute trained-model inference | Not established | Qualified local/remote computation over actual model parameters |
| Use a provider without local weight download | Possible future mode | Authorized inference endpoint, model/version pin, input/output contract and checked receipt |
| Produce equivalent output by state navigation | Research goal | Evidence that the navigator realizes the same conditional computation, not a shape-only similarity |

An external inference provider could hold the weights on its own hardware while VL-64 sends requests and checks results. That avoids *downloading* weights to the local client, but does not eliminate the physical computation or parameter access performed by that provider. A procedurally reconstructible exact weight Ground would need a separate proof of its adequacy.

**Never return a synthetic/randomly initialized model output as though it came from the trained model.**

## What the implementation already proves

The November 18 GitHub Actions execution [run 44387089534](https://github.com/Kier73/VL-64/actions/runs/44387089534) successfully processed observer-selected `json`, `text`, and `jsonl` requests. In that run:

- a current recent-catalogue selector returned a partial Ground for a repository without usable architecture;
- the pinned GPT-2 model returned actual declared config in human-readable form;
- the GitHub document intention resolved a witnessed network path and returned a machine-readable receipt;
- no weights were downloaded and no trained-model inference was claimed.

The run passed 57 tests at that point, covering the compact contract, output format, fixed registered operations, target validation, metadata-only refusal, and absence of arbitrary inference execution.

The [second measured session](https://github.com/Kier73/VL-64/actions/runs/44389239409) added an observer constraint requiring architecture metadata from a Transformers-filtered recent catalogue. Its selector attempted several recent models and accepted the fourth, which exposed `model_type`, `vocab_size`, and `head_dim` fields. The evidence and refusal trail are retained as a [research dataset](../experiments/observer_intent/datasets/2026-11-18-run-44389239409-summary.json). The result remained an **incomplete architecture for model inference**, despite meeting the narrower observation predicate.

This is a **live test of observer intention arbitration**, not a complete self-adapting network computation compiler.

## Next meaningful interfaces

1. **Typed outcome predicates.** Beyond requiring an observed config, support `must_include`, acceptable model family, revision age, max transfer bytes, and a target observable.
2. **General Ground route catalogue.** Let adapters advertise typed operations as source/target representation relations with declared side effects, freshness, cost and verifying witnesses.
3. **Independent planning and execution.** Separate what the observer wants from service selection, realizer admission, data acquisition, and result checking. Plans must fail closed on stale evidence or incompatible representations.
4. **Model structure descriptors.** Build versioned typed operator DAGs for supported model families from config and tensor-header metadata. Use bounded range reads where justified, without downloading tensor values; preserve the file/offset evidence and incomplete shapes.
5. **Remote model execution as a distinct operation.** After explicit user authorization and an available provider, build a safe `hf.model.infer` with actual inference receipts, input/output validation, strict cost and token controls, and pinned model identity. Do not call inference when the request only asks for a structure.
6. **Feedback and adaptation.** Compare failed requests, new candidate Grounds, completed observations, and measured costs. Discover a better representation or path only after it improves the validated objective.
7. **Portable communication.** Export typed intentions, Grounds and receipts as small canonical packets usable across host processes and network paths. Preserve what evidence was produced locally versus what remote sources actually supplied.

The desired end state is a **communicable, substrate-independent VL arbitrator** that can answer an observer's request by selecting the smallest sufficient combination of existing computational Grounds, qualified operations, and evidence—not a system that must iterate over everything the external system can potentially do.
