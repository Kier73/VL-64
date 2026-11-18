# Generic Byte Fabric — External Observer to Qualified Realizer

**Purpose:** make the observer's request **architecture-, source-, webpage-, and data-type-independent** at the representation layer. The same contract accepts arbitrary bytes and preserves their exact observed identity, even when no decoder recognizes their meaning. Specialized interpretation and realization are explicit, optional host capabilities.

This lab is outside the 65,536-byte VL-64 source package, reusing the existing `vld.ObserverIntention` contract and Ground digests. It does **not** claim arbitrary machine instruction execution or full semantics from a signature.

## Core idea

~~~text
external observer intent (operation, target, options, output format)
  → source capability selection
  → bounded, provenance-labeled byte acquisition
  → exact byte Ground + optional typed views
  → optional qualified pure operator
  → result + execution receipt + replay checks
~~~

This works whether the source is a local file, an approved HTTPS page, an inline byte sequence, or a host-installed sensor/process/screen/stream adapter. A binary that has never been encountered is still valid input.

## Ask a host what it can do

The observer can query the host's installed capabilities **without knowing its architecture**:

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/host_capabilities.json \
  --out available-capabilities.json
```

The returned catalogue lists byte sources, built-in and registered decoders, typed local operators, input/completeness requirements and permitted HTTPS hosts. A new observer-host can advertise its own sources or operators without changing the compact VL-64 core.

## Describe the result you want, not its implementation

The `bytes.realize` intention asks for a **result kind** (such as `value.json`, `digest.sha256`, `links.html`, or a host-defined kind) rather than a named operator. A [GoalNavigator](navigator.py) chooses among installed relations, checks source completeness and typed-view requirements, and returns a replayable path-and-execution receipt. The host can register new relations without modifying the compact VL-64 core.

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/goal_json_value.json \
  --out selected-result.jsonl
```

See the [goal-directed navigation design](../../docs/GOAL_DIRECTED_BYTE_NAVIGATION.md) and the live [goal-navigation workflow](../../.github/workflows/goal-navigation.yml).

## General callable operations

| Observer operation | Result | Requirements |
| --- | --- | --- |
| `bytes.observe` | Exact byte/prefix digest, completeness, format hints, Ground | A qualified source adapter |
| `bytes.execute` + `sha256` | SHA-256 of **all** acquired bytes | Complete source |
| `bytes.execute` + `slice` | Base64-encoded exact byte interval | Interval wholly observed; up to 4 KiB return |
| `bytes.execute` + `json.pointer` | JSON scalar/object/array at requested RFC 6901-like pointer | Complete, valid JSON |
| `bytes.execute` + `html.links` | Syntactically observed page links and title | Complete, valid UTF-8 HTML |
| `bytes.execute` + `bytes.stats` | Count, zero bytes, distinct-byte count | Partial accepted, but result clearly scoped to captured bytes |

Only *registered* operators may execute. Byte-observation and actual action authority are separate. There is no `python.eval`, arbitrary system command, unrestricted API POST, or implicit model-inference operator.

## One contract for every observed system

~~~json
{
  "operation": "bytes.execute",
  "target": "workspace:experiments/byte_fabric/fixtures/state.json",
  "output": "jsonl",
  "options": {
    "operator": "json.pointer",
    "parameters": {"pointer": "/observer/constraints/verified"}
  }
}
~~~

Run from the repository checkout:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/json_execute.json \
  --out output.jsonl
~~~

The result contains the submitted intention and its digest, a declared plan, a Ground digest, the exact source/prefix digest and declared format, and a separate deterministic execution receipt.

### Live arbitrary page

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/live_webpage.json \
  --out page-ground.json
~~~

This sample observes `https://example.com/` as bytes, then recognizes HTML. It does not need a hardcoded example.com semantics module.

To permit a **new trusted public HTTPS host**, explicitly pass `--allow-host host.example` to the CLI, rather than letting an untrusted intention grant itself network access. The observer request only describes a destination; the host running the adapter sets its own scope. DNS public-address filtering is a defense-in-depth check; it is not an IP-pinned anti-rebinding security guarantee. Avoid registering untrusted domains where local network access must be excluded by policy.

### Arbitrary file and opaque binary

~~~json
{
  "operation": "bytes.observe",
  "target": "inline:b64,/wECAwQF//4=",
  "output": "json",
  "options": {"include_data": true}
}
~~~

`include_data=true` returns up to 4 KiB of observed content as base64 alongside its digest and clear full/prefix scope. The same option works on a file or web page. For larger observations, use a registered qualified slice operation or an external streaming/capture adapter.

Local `workspace:` paths are resolved under an explicit `--byte-root` (default current working directory). Path escapes are refused, including traversal and symlink resolutions outside the root. The source is never authorized to execute simply because its extension resembles a program.

## Portable built-in recognition

| Data family | What is described | What is **not** claimed |
| --- | --- | --- |
| Arbitrary unknown bytes | Exact content/prefix digest, byte count, sample | File type or computation semantics |
| UTF-8 text | Text statistics and prefix | Meaning, truth, embedded code authority |
| JSON | Container, keys, length and supported pointer projections | Authenticity or semantics of arbitrary fields |
| HTML | Title, discovered links | Browser execution, navigation permission or rendered layout |
| PNG / GIF | Header dimensions | Pixel decode, correct CRC, full file validity |
| JPEG / PDF / ZIP | Signature or version hint | Complete format parsing or safe extraction |
| ELF / WASM / MZ | Header class/endian/version hints | Verification or execution of machine instructions |
| Safetensors | Bounded JSON header, dtype/shape/offset sample | Trained tensor payload, model behavior or weight reconstruction |

A format detection is a *declared interpretation based on observed bytes*, never proof that the file is a semantically valid executable, image, PDF, or model.

### Inspecting Safetensors without reading weight values

When the source hint ends in `.safetensors`, the lab can parse the first eight bytes as the little-endian JSON header length, then inspect only that header if it fits the capture budget. It reports tensor names, shapes, dtypes and data offsets, with `weight_payload_interpreted=false`. Header interpretation is based on the public [Safetensors format specification](https://github.com/huggingface/safetensors#format); HTTP range transfer may supply only the relevant bytes. If the header is not completely acquired, the Ground says `safetensors-unobserved-header` rather than producing invented tensor properties.

For an actual remote Safetensors file, hosting/CDN and redirect policy must explicitly authorize the header-range transfer. The default HTTPS adapter refuses unapproved cross-host redirects and does not download weights in the preconfigured tests.

## Host plug-ins are the generality mechanism

The optional [extensions.py](extensions.py) registry supports additional systems **without special-casing them in VL-64**. A host can register a source such as `sensor:`, `process:`, `screen:`, `packet:`, `memory:` and a typed decoder. The source callback returns bytes plus explicit completeness/provenance; the decoder adds a view while the original byte identity remains authoritative.

~~~python
from vld import ObserverIntention
from experiments.byte_fabric.sources import capture_bytes
from experiments.byte_fabric.extensions import (
    SourceRegistry, SourceCapability, DecoderRegistry)
from experiments.byte_fabric.router import byte_arbitrate

sources = SourceRegistry()
sources.register(
    SourceCapability("sensor:"),
    lambda identifier, limit: capture_bytes(b"\x49\x00\x00\x00", identifier)
)
decoders = DecoderRegistry()
decoders.register(
    "u32-le",
    lambda raw, hint: len(raw) == 4,
    lambda raw, hint: {
        "interpreted_register": int.from_bytes(raw, "little"),
        "declared_encoding": "unsigned-little-endian-32"
    },
)
request = ObserverIntention(
    "bytes.observe", "sensor:counter", options={})
receipt = byte_arbitrate(request, sources=sources, decoders=decoders)
assert receipt["status"] == "OBSERVED_BYTE_GROUND"
assert receipt["ground"]["extended_views"][0]["value"]["interpreted_register"] == 73
~~~

Only host-side installation registers callbacks. Arbitrary remote bytes, a JSON intention, a webpage or a model card cannot install executable Python code through this interface.

A `QualifiedByteExecutor` can likewise register a new pure local operator under an explicit `OperationCapability`. The generic router considers only the admitted registry; other requests are refused. Model inference, remote writes, arbitrary code execution and spending must use **separate authority-bearing realizers** with independent verification.

## Demonstrated, independently tested paths

On a real GitHub Actions Linux runner, the lab observed an ordinary public webpage (577 bytes), the pinned VL-64 README (7,832 bytes), an explicitly incomplete 512-byte web prefix, arbitrary binary input, a local JSON structure, and an HTML fixture. It executed JSON-pointer extraction, HTML link extraction and SHA-256 over a complete acquired web response. Both Python 3.11 and 3.13 passed the initial 95-test suite.

The tests additionally supply synthetic byte headers for ELF, WASM, PDF, PNG, GIF, ZIP, JPEG and Safetensors, confirming that the same Ground API accommodates architecture-specific views and an opaque fallback. The synthetic files validate recognition behavior, not the validity of a full execution architecture.

The laboratory also acquired an **actual 8,991-byte PNG** from a public GitHub repository through the same generic HTTPS source as HTML, text and JSON. The recognised PNG Ground exposes declared dimensions without decoding the image or executing embedded data.

The [versioned public observation dataset](datasets/2026-11-18-run-44397602865-summary.json) preserves nine source/interpretation/execution summaries and independently recorded **matching structural Ground digests across Python 3.11 and 3.13**. These are execution logs and content-derived identifiers, not external attestations.

The workflow [generic-byte-fabric.yml](../../.github/workflows/generic-byte-fabric.yml) saves complete result receipts for inspection. More recent runs may have additional tests.

## Source and execution safety rules

- Source acquisition: explicitly authorized local root, bounded inline bytes, or HTTPS host allowlist with public-DNS check.
- HTTPS: no redirects, URL credentials, private addresses or unsafe ports; range responses require exact `Content-Range` consistency, and a server ignoring Range is handled as a bounded prefix when necessary.
- Capture: content and completeness are separate; a prefix digest is **never** a full-resource digest.
- Decoder: has no authority to fabricate unobserved bytes or override exact source identity.
- Execution: a registered, pure, bounded operation; full-file operations refuse incomplete captures.
- Receipts: bind input Ground, operation and result; local replay recomputes the result when given original bytes. This does not establish external authority, absolute truth or a trusted remote execution.
- User output: JSON, JSONL and text, with binary results represented as explicit base64; raw binary streaming and arbitrary output schemas remain future adapters.

## Compressive, internally navigable Grounds

The [compressive Ground module](orbit_ground.py) discovers a changing scalar span from byte snapshots, tests candidate endian/decimal interpretations and an affine transition law against independent held-out frames, then retains a **template plus law** rather than all states.

A [four-hop typed path](../../docs/COMPRESSIVE_VIRTUAL_GROUNDS.md) interprets a saved frame, navigates the modeled state using cached powers, reconstructs a requested frame and derives its digest. The saved [Ground](datasets/2026-11-18-compressive-orbit-ground.json) is usable with the observer's own output format:

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/ground_trillion.json \
  --out result.json
~~~

The measured fixed-layout experiment encoded 81,920 bytes of training and validation snapshots with a 1,947-byte description (42.075×). A separate challenge API checks new observed frames and returns explicit contradictions without silently rewriting the Ground. This experiment is **a validated state generator for one pattern**, not arbitrary machine emulation.

## Non-affine and observer-specific Virtual Grounds

The [non-affine byte Ground](discrete_ground.py) learns a finite deterministic preperiod/cycle from observed byte snapshots and validates exact reconstruction on held-out frames. [GroundFamilies](ground_families.py) compares the finite interpretation with the modular-affine interpretation, selecting the smallest qualified stored description.

The [observable quotient](quotient_ground.py) reuses the compact core's Myhill–Nerode partition refinement to retain only states distinguishable by the observer's declared outputs and future transitions. It is deliberately **not a lossless Ground** for discarded source bytes.

Two independently navigable examples:

```bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/nonaffine_trillion.json \
  --out exact-modeled-result.json
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/observable_quotient_trillion.json \
  --out observable-only-result.json
```

See the [evidence and methodology](../../docs/NONAFFINE_AND_OBSERVABLE_GROUNDS.md), [saved non-affine Ground](datasets/2026-11-18-nonaffine-ground.json), [saved quotient](datasets/2026-11-18-observable-quotient-ground.json) and [controlled measurements](datasets/2026-11-18-nonaffine-and-observable-measurements.json). These experiments change no source bytes in the compact runtime.

## Active Ground experiments

The [active Ground navigator](active_ground.py) and [host-facing experiment](active_lab.py) extend static state inference into **intention-driven observation**. Candidate finite action models predict exact byte-output traces. A bounded planner selects the most informative affordable action word; a separately admitted resettable host probe supplies real outputs, and the observations eliminate inconsistent descriptions. The candidate set is never silently replaced when every model is contradicted.

```bash
python -m experiments.byte_fabric.active_lab --out artifacts
python -m experiments.observer_intent.arbitrate \
  --intent artifacts/active-next-probe-intent.json \
  --out artifacts/suggested-experiment.json
```

A separate optional `--live-only` experiment uses **actual read-only public GitHub bytes** for a resource-identity discrimination task, not a remote stateful model. See [Active Action-Conditioned Grounds](../../docs/ACTIVE_ACTION_GROUNDS.md) for its authority and evidence contracts.

## Synthesize the interpreter from observed actions

[TraceGroundSynthesizer](trace_synthesis.py) constructs small finite deterministic Ground hypotheses directly from recorded resettable action/output traces. Unlike the previous [active Ground module](active_ground.py), the observer does not need to supply candidate models. A bounded prefix-tree state-merging search proposes descriptions; actual host-qualified observations update the evidence ledger and cause a fresh synthesis.

```bash
python -m experiments.byte_fabric.trace_lab --out artifacts
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/trace_derived_next_probe.json \
  --out next-experiment.json
```

The [saved trace ledger](datasets/2026-11-18-initial-trace-ledger.json) reproduces candidate synthesis, and the [saved Ground](datasets/2026-11-18-trace-derived-action-ground.json) navigates the learned state model. See the [method and measured evidence](../../docs/TRACE_DERIVED_ACTION_GROUNDS.md). A separate optional experiment probes pinned GitHub files under read-only authorization; it does not claim those static files form a stateful service.

## Next frontier

The general observer-to-realizer interface should evolve into an **evidence- and cost-aware typed relation search**:

1. User describes a desired observable or state transition, not the exact URL and interpreter.
2. A catalogue of registered sources, decoders and realizers advertises input/output types, side effects, evidence expectations and budget.
3. A planner chooses a valid composition and requests the smallest sufficient observations, preserving partial and unknown states.
4. A verifier checks the result against independently grounded evidence; expensive or side-effecting operations require explicit authorization.
5. A newly learned Ground may offer a cheaper equivalent realization only when that equivalence survives independent checks.

This applies equally to source files, page content, device registers, model structures, communications and other byte-bearing environments. **Universal representation is already available; universal interpretation and verified execution remain research and adapter-extension tasks.**
