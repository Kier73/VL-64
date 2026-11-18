# Observer Intention Laboratory

VL-64 can now accept a **typed external observer intention** and delegate it to a qualified, locally registered observation operator. This laboratory binds the compact \`vld.ObserverIntention\` contract to public read-only Hugging Face and GitHub metadata resources. The larger network adapter lives **outside** the 64 KiB VL-64 source budget.

The observer determines *what* to observe and in which output representation. The adapter determines which read-only operations are available, what it can observe, and what claims it can justify.

## Input contract

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

The exact \`operation\` is a registered operator name, **not natural language**. The \`target\` is a model identifier or an explicitly supported catalogue selector. The \`output\` is one of \`json\`, \`jsonl\`, or \`text\`. The \`options\` is a bounded JSON object.

The compact runtime validates this contract and binds its contents to \`intention_digest\`. The external adapter compiles it into an explicitly enumerated read-only plan and emits its own \`plan_digest\`. Neither digest authenticates the remote service.

## Run from a clone

~~~bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/
python -m experiments.observer_intent.arbitrate \
  --intent experiments/observer_intent/intents/hf_latest_structure.json \
  --out hf-latest-structure.json
python -m experiments.observer_intent.arbitrate \
  --intent experiments/observer_intent/intents/hf_known_structure.json \
  --out hf-reference-structure.txt
python -m experiments.observer_intent.arbitrate \
  --intent experiments/observer_intent/intents/github_research_goals.json \
  --out github-document-ground.jsonl
~~~

PowerShell may use the same \`python -m\` commands on one line. A source checkout is required for the experimental module; the \`vld\` package alone exposes the compact intention contract.

## Operators currently supported

| Operator | Input \`target\` | Actual operation |
| --- | --- | --- |
| \`hf.model.describe\` | A public \`namespace/repository\` identifier | Read bounded model information, list available filenames, and read pinned \`config.json\` when present |
| \`hf.model.describe\` | \`latest:text-generation\` | Query a bounded text-generation catalogue ordered by **last modification**, select its first accessible entry, then inspect its metadata |
| \`github.document.describe\` | \`Kier73/VL-64:docs/RESEARCH_GOALS.md\` | Use the existing GitHub network lab to locate the pinned README and requested linked document |

Unknown or side-effecting operations are **unresolved**. Requests for actual model inference do not silently substitute fake or randomly initialized results.

The current \`latest\` selector reports the **first eligible entry within the bounded most-recently-modified catalogue response**. It is not the latest newly published foundation model and not necessarily an architecture with a usable config. If that entry lacks \`config.json\` the outcome is \`PARTIAL_MODEL_STRUCTURE\` and the missing architecture is explicitly recorded.

## Evidence and output

A successful Hugging Face observation includes:

- original observer intention, intention digest, selected operator, and plan digest;
- model repository ID and pinned commit revision;
- explicit source URLs, response hashes, byte counts, timestamps and truncation/refusal statuses;
- declared architecture/config fields if a pinned \`config.json\` was accessible;
- listed model weight filenames and reported Safetensors summary, if available;
- a digest-bound Virtual Ground with metadata, revision, configuration and weight-manifest nodes;
- explicit claims that **zero weight payload bytes were read**, no model inference occurred, and no arbitrary model code was executed.

The \`json\` output is a readable JSON receipt. \`jsonl\` is one machine-readable receipt per line. \`text\` is a concise human-readable inspection. The observer chooses the output; an adapter cannot change it silently.

A \`PARTIAL_MODEL_STRUCTURE\` is an outcome that preserves real observations and unknowns. An \`OBSERVED_MODEL_STRUCTURE\` means architecture metadata was read, **not** that the trained model's semantics or parameters were reconstructed or executed.

## Executed results

On November 18, 2026, the [GitHub Actions run](https://github.com/Kier73/VL-64/actions/runs/44387089534) demonstrated all three paths:

- The live catalogue chose a recently modified public text-generation repository; it did not expose a usable \`config.json\` and was reported as partial.
- The pinned control \`openai-community/gpt2\` returned observed architecture/config metadata, in \`text\` format, from revision \`607a30d783dfa663caf39e06633721c8d4cfcd7e\`.
- The GitHub document operator navigated the repository's pinned document link and emitted \`jsonl\`.
- **57 tests passed** before subsequent additional security tests.

The workflow attaches the generated outputs as artifacts for independent inspection.

## What this enables, and what comes next

The external observer now has a first callable interface for describing work *without specifying every network hop*. The missing generalization is to promote the operator registry into an evidence-qualified planner over typed Ground relations, with freshness and cost constraints.

For model execution in particular, architectural metadata can describe tensor shapes and an operation graph, but **does not contain the trained parameter values**. To produce the trained model's actual outputs without downloading weights, VL-64 would need a qualified remote inference observer-host, a provider model/version agreement, credentials and costs where needed, and independently checked input/output receipts. That is a separate realization mode from reading model structure.

Further directions:
1. Generate and compare typed operator graphs from known model-family configurations, distinguishing declared metadata from inferred architecture.
2. Add explicit outcome predicates such as “architecture must be present” instead of silently returning an inadequate candidate.
3. Qualify alternative implementations of the same intention (metadata API, pinned config, Safetensors header metadata, remote inference service), choosing with evidence/cost rules.
4. Admit bounded remote inference only with explicit authority, provider qualification, model revision, and spend limits; never infer a numeric result from metadata alone.
5. Convert successful portable observer contracts into a more general [network-native 64 KiB design](../../docs/VIRTUAL_INTERNET_RUNTIME_PLAN.md).

For live network datasets and their existing evidence checks, see [the Internet Ground laboratory](../internet_lab/README.md).
