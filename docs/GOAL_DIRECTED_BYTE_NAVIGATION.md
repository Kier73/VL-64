# Goal-Directed Byte Navigation

**Implementation:** [GoalNavigator](../experiments/byte_fabric/navigator.py) with \`vld.ObserverIntention\` and [Generic Byte Fabric](../experiments/byte_fabric/README.md).  
**Status:** Executable first pass, not a general-purpose AI inference or arbitrary-program engine.  
**Runtime budget:** Goal search lives in a host adapter. The existing compact \`vld\` package remains below 65,536 bytes.

## The observer describes the result, not the operator

Previously an observer needed to name a particular realization method, such as \`json.pointer\` or \`sha256\`. The new operation, \`bytes.realize\`, lets the observer identify **the desired result type**. A host-installed relation catalogue selects among qualified operations that produce that result.

~~~json
{
  "operation": "bytes.realize",
  "target": "workspace:experiments/byte_fabric/fixtures/state.json",
  "output": "jsonl",
  "options": {
    "goal": {
      "kind": "value.json",
      "parameters": {"path": "/observer/constraints/verified"},
      "expected_result": {"value": true, "value_type": "bool"}
    }
  }
}
~~~

The result includes the selected relation (\`json-field\`), the actual operator (\`json.pointer\`), the captured byte Ground, admission checks, a local execution receipt, and a replayable navigation plan.

The user is not asking VL-64 to execute arbitrary natural language. The small typed result vocabulary is the observer-facing language. A natural-language assistant can eventually generate these intentions, but a host must still check the typed request and its authority.

## General result types

| Requested kind | Possible qualified operator | Requirements |
| --- | --- | --- |
| \`digest.sha256\` | \`sha256\` | Complete byte resource |
| \`bytes.interval\` | \`slice\` | Requested interval within observed bytes |
| \`value.json\` | \`json.pointer\` | Complete resource with verified JSON syntax |
| \`links.html\` | \`html.links\` | Complete UTF-8 HTML observation |
| \`stats.bytes\` | \`bytes.stats\` | May operate on a separately marked prefix |
| Host-defined result kind | Registered host operation | Explicitly qualified input view and capability |

All kinds pass through the same external observer intention, independent of whether the input is a web page, model metadata, image, binary file, screen buffer or sensor frame.

### Arbitrary architecture and machine sources

A host can register its source, decoder and pure local operator without modifying VL-64. For example, \`sensor:\` bytes may be interpreted by a registered \`register32\` view and a host-provided \`read.sensor\` operator. The host can advertise a new relation

~~~python
GoalRelation(
    name="read-sensor-u32",
    produces="value.sensor",
    operator="read.sensor",
    requires_view="register32",
    estimated_work=4,
)
~~~

and an external observer can ask for \`value.sensor\` without knowing that the host uses a particular decoder or operator. This depends on explicit host installation; untrusted input cannot install a realizer.

## How navigation works now

1. **Compile** an observer intention, check its goal syntax, ensure the host advertises at least one relation producing the desired result. Unsupported requests are unresolved *before any resource acquisition*.
2. **Observe** a permitted byte source with exact or explicitly partial evidence.
3. **Construct Ground** over the captured bytes plus any qualified optional interpretations.
4. **Select** from matching relations whose input-view, completeness, capture size and estimated-work conditions hold. The planner records qualified and rejected candidates, not just the winner.
5. **Execute** the least-estimated-work candidate first, then any remaining permitted pure local candidates if the first cannot satisfy the request. Never improvise a remote side-effecting action.
6. **Check** the result against an optional exact expected result. A mismatch is not accepted, even if an operation returned data.
7. **Return** in observer-selected JSON, JSONL or text; retain a Ground digest, observation digest, plan digest, and execution receipt.
8. **Replay** with the original bytes and installed qualified capabilities, recomputing the route and result. Any altered result or selected path is rejected.

The estimates currently represent *host-declared static work units*, not measured wall time or a calibrated optimizer. Choosing lower estimated work is an explicit policy whose accuracy must be tested separately.

An observed relation catalogue is also not execution authority. It may be used only if the host has actually registered the corresponding qualified pure operator.

## What changes for the external observer

A page can be requested through a goal to return links, bytes, statistics or a digest. A Hugging Face model API response is treated as generic JSON bytes; a request can ask for \`value.json\` at \`/id\` without requiring the Hugging Face-specific architecture adapter. A PNG can yield a byte digest without pixel decoding. An unfamiliar machine binary can still provide an exact byte interval, even if no executable semantics have been discovered.

The desired output type is chosen first. The host chooses *how* to realize it based on what it can actually interpret and compute.

### Run the first experiments

From a VL-64 checkout:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_goal_navigation.py
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/goal_json_value.json \
  --out selected-result.jsonl
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/goal_web_digest.json \
  --out web-result.json
~~~

Additional examples include \`goal_model_id.json\` (generic Hugging Face JSON), \`goal_png_digest.json\` and \`goal_unknown_slice.json\`. The [goal-navigation workflow](../.github/workflows/goal-navigation.yml) records live observer-selected results and checks their selected relations.

## Limitations and the next real ambition

This first implementation selects **one qualified operation over one observed source**. It is a meaningful upgrade from hardcoding the operator, but it is not yet arbitrary multi-step state-space computation.

Next investigations should implement:
- **Multi-hop relation composition:** select among \`source → acquired bytes → parsed type → transformed state → desired result\` paths, with compatible edge types and independently checkable evidence at each hop.
- **Real measured costs:** compare transfer bytes, parsing overhead, work, verification and recovery, and learn when cached/structural paths outperform a fresh request.
- **Network execution admission:** allow a remote realizer only when an independently defined input/output, model/version, authority and cost contract is satisfied.
- **Typed result refinement:** goals should support observable predicates, shape constraints, semantic equivalence and controlled state transitions rather than only single result kinds.
- **Distributed verification:** move an intention, Ground and output receipt across observer-hosts, reproduce the result on another qualified implementation and record any disagreement.
- **Architectural reconstruction:** extract and validate deeper relations from arbitrary machine and model representations. Byte equality and syntactic header parsing do not prove that a virtual execution is semantically equivalent to the original physical/software system.

The falsifiable target remains: **an observer names the outcome; the VL selects an eligible computation path; the outcome and evidence survive independent reconstruction at lower total measured work.**
