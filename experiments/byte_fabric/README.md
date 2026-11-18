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

The workflow [generic-byte-fabric.yml](../../.github/workflows/generic-byte-fabric.yml) saves complete result receipts for inspection. More recent runs may have additional tests.

## Source and execution safety rules

- Source acquisition: explicitly authorized local root, bounded inline bytes, or HTTPS host allowlist with public-DNS check.
- HTTPS: no redirects, URL credentials, private addresses or unsafe ports; range responses require exact `Content-Range` consistency, and a server ignoring Range is handled as a bounded prefix when necessary.
- Capture: content and completeness are separate; a prefix digest is **never** a full-resource digest.
- Decoder: has no authority to fabricate unobserved bytes or override exact source identity.
- Execution: a registered, pure, bounded operation; full-file operations refuse incomplete captures.
- Receipts: bind input Ground, operation and result; local replay recomputes the result when given original bytes. This does not establish external authority, absolute truth or a trusted remote execution.
- User output: JSON, JSONL and text, with binary results represented as explicit base64; raw binary streaming and arbitrary output schemas remain future adapters.

## Next frontier

The general observer-to-realizer interface should evolve into an **evidence- and cost-aware typed relation search**:

1. User describes a desired observable or state transition, not the exact URL and interpreter.
2. A catalogue of registered sources, decoders and realizers advertises input/output types, side effects, evidence expectations and budget.
3. A planner chooses a valid composition and requests the smallest sufficient observations, preserving partial and unknown states.
4. A verifier checks the result against independently grounded evidence; expensive or side-effecting operations require explicit authorization.
5. A newly learned Ground may offer a cheaper equivalent realization only when that equivalence survives independent checks.

This applies equally to source files, page content, device registers, model structures, communications and other byte-bearing environments. **Universal representation is already available; universal interpretation and verified execution remain research and adapter-extension tasks.**
