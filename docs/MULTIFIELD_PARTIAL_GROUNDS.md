# Multi-Field and Partial-Observation Virtual Grounds

**Status:** Executable controlled experiment.  
**Runtime:** 65,093 bytes / 65,536-byte source budget; the implementation is external to `src/vld/`.  
**Components:** [multifield_ground.py](../experiments/byte_fabric/multifield_ground.py), [multifield_adapter.py](../experiments/byte_fabric/multifield_adapter.py), [multifield_lab.py](../experiments/byte_fabric/multifield_lab.py).

## Why this follows the monolithic seed

The [monolithic seed](MONOLITHIC_SEED_AND_SELF_GROUNDS.md) demonstrates that VL-64 can procedurally construct an exact source-level Ground of its own modules and relations. The next question is whether it can **learn richer Grounds from external evidence that is only partially visible**.

Earlier byte-state experiments required one contiguous changing span. That is too restrictive for packet headers, memory images, multiple registers, video buffers, software process state and composed observer-host descriptions, where logically related variables may occur at widely separated byte offsets.

This experiment learns:

- A **fixed byte context**, represented once.
- Multiple **disjoint changing byte spans**, discovered directly from observed before/after frames.
- A **joint finite state** formed by concatenating the actual varying bytes, without assuming each region evolves independently.
- A **sparse action-conditioned transition graph** that contains only witnessed transitions. Missing edges are explicit unknowns.
- **Witnessed conditional sensitivity relations** when observations differ in one input span under the same action and corresponding output spans differ.
- A **partial-observation navigator** that compares only supplied byte intervals and asks which additional span would best distinguish compatible modeled states.

The Ground is a structural description and an admitted finite observed transition model, not a universal reverse-engineered instruction set.

## Ground formation from raw bytes

Training observations have the shape:

~~~python
(before_frame_bytes, "qualified.action", after_frame_bytes)
~~~

The learner checks that every frame has a consistent declared width and searches for positions where any witnessed state differs from the first frame. Adjacent changed positions form a span. All other bytes are retained in a single static template. The state token for an observation is the concatenation of just the variable spans.

For two distant fields, a frame of length 512 could be represented as:

~~~text
0                                                             511
|      fixed bytes     |mode|       fixed bytes       |counter|fixed|
         context      offset 73                  offsets 311–312
~~~

The discovered Ground stores two variable-span descriptors, one reusable template and the exact variable bytes for each witnessed state. Each action edge refers to admitted state identities.

An action applied to an unknown state, or a known state/action pair not present in the observed edge table, remains **UNRESOLVED**. No assumed value is inferred from the nearest state, and a held-out observation with an unseen edge is marked unresolved rather than treated as a successful prediction.

## Partial bytes are first-class observations

An external observer does not need to transmit a complete 512-byte frame simply to ask which modeled state might be present.

The new `ground.multifield.observe` request carries a list of **known intervals and the exact bytes for those intervals only**:

~~~json
{
  "operation": "ground.multifield.observe",
  "target": "workspace:experiments/byte_fabric/datasets/2026-11-18-multifield-ground.json",
  "output": "json",
  "options": {
    "goal": {
      "known_intervals": [[73, 74]],
      "segments_b64": ["AQ=="]
    }
  }
}
~~~

The one-byte input is sufficient to constrain the model's **mode** but does not identify the counter. VL-64 reports four compatible modeled states, then selects the missing two-byte counter span, offsets `[311, 313)`, as the best next observation.

The planner computes partition information under a *uniform prior over compatible model states*. Its value of **2 bits** is a model-relative expected distinction, not an externally measured Shannon entropy of all possible machine behavior. The observation remains partial even if the model finds a unique candidate: unobserved bytes have not thereby been independently checked.

If a supplied observed byte contradicts the constant context or every modeled state, the result is `CONTRADICTED`. If multiple states remain indistinguishable under available fields, it returns `INDETERMINATE_COMPATIBLE_STATES`.

## Navigate the joint state space

The action-conditioned Ground stores transitions over the **combined multi-field state**. This makes it possible to retain interactions—for example, a `tick` action that increments a counter by different amounts depending on the current mode.

Two navigation forms are now implemented:

- **Repeated action:** `ground.multifield.navigate` with `action` and `steps`; traverse only observed edges and skip a verified deterministic model cycle when revisiting a known state.
- **Mixed action word:** the same observer operation with an explicit `actions` array; compose the witnessed state/action edges in order, preserving evidence for every transition.

Both use a compact state token to identify the starting state, rather than requiring a complete frame input. The full frame can be reconstructed from the Ground's static template and the terminal state's variable spans.

### Three callable observer intentions

~~~bash
python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_recommend_field.json \
  --out suggested-byte-observation.json

python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_trillion.json \
  --out modeled-distant-state.json

python -m experiments.observer_intent.arbitrate \
  --intent experiments/byte_fabric/intents/multifield_mixed_program.json \
  --out composed-transition-result.json
~~~

The first returns a proposed observation. The second computes a model's distant-state digest. The third navigates a bounded mixed action sequence. **None** executes an action on an external service. Actual acquisition and actuation remain separate host-authorized operations.

## Measured controlled experiment

A 512-byte static context was taken from the actual VL-64 `src/vld/__init__.py` file. The experiment replaced two separated byte regions with state variables generated by an **independent synthetic reference transition function**. This is a meaningful structural-learning test but is not evidence that VL-64 source itself operates as that dynamic machine.

| Measurement | Observed result |
| --- | ---: |
| Input frame width | 512 bytes |
| Changed spans discovered | offset 73 (1 byte); offset 311 (2 bytes) |
| Distinct modeled states | 8 |
| Registered actions | 2 |
| Witnessed state/action edges | 16 |
| Training transition frame bytes | **16,384** |
| Compact multi-field Ground | **4,322 bytes** |
| Training-to-description ratio | **3.791×** |
| Checked additional observations of known edges | 16 |
| New mixed action words checked against independent reference | **62** |
| Partial known bytes used in one intention | **1** |
| Compatible states after that observation | 4 |
| Recommended next observation | bytes 311–312 |
| Predicted information gain (uniform model states) | **2 bits** |
| Navigated modeled horizon | **1,000,000,000,003** |
| Model transitions individually evaluated | **7** |
| Modeled transitions skipped through a witnessed cycle | **999,999,999,996** |

The 62 mixed action words were **new sequences assembled from already witnessed state/action edges**. They test compositional correctness of the learned Ground, not the discovery of unobserved transition outcomes.

A deliberately withheld `state/action` pair caused an explicit `UNRESOLVED` response, even for the trillion-step request. A changed static byte produced a `CONTRADICTED` challenge receipt. The original Ground and its evidence remained unchanged.

The dependencies are *witnessed sensitivities*: when two observed inputs differed in one span under the same action, and their outputs differed in another span, the Ground recorded the relation. Such observational comparisons are not by themselves a controlled causal proof.

## Reproduce the experiment and retain evidence

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.byte_fabric.multifield_lab --out artifacts
~~~

The [stored multi-field Ground](../experiments/byte_fabric/datasets/2026-11-18-multifield-ground.json), [saved measurements](../experiments/byte_fabric/datasets/2026-11-18-multifield-measurements.json) and the three [typed example intentions](../experiments/byte_fabric/intents/) are versioned for independent replay. The [multi-field CI workflow](../.github/workflows/multifield-ground.yml) runs the full repository suite on Python 3.11 and 3.13, performs the actual experiment and preserves all produced outputs.

## Generality and the next frontier

This experiment uses a **fixed-length, byte-indexed frame** with a bounded deterministic finite state graph. It cannot yet infer arbitrary variable-length messages, arbitrary CPU instruction semantics, a large neural model's forward pass or hidden effects in a remote network service.

The next research steps are:

1. **Variable-length and relocatable fields:** parse framing, length-prefixes, symbolic offsets and structured relocations so Grounds are not tied to byte positions alone.
2. **Sparse relational laws:** learn independently verifiable arithmetic, conditional and symbolic dependencies between fields to avoid storing every combined state as a table.
3. **Partially observed transitions:** admit before/action/after records where only selected byte regions were acquired, preserving three-valued observation logic rather than requiring complete frames to learn any edge.
4. **Counterexample-guided model revision:** retain old Ground witnesses and construct a new Ground when later observations falsify a relation.
5. **Direct HTTP, DNS and Git observers:** use permitted resettable/read-only host adapters to collect real request/response traces and determine which fields are stable, which vary and which transitions are actually supported.
6. **Seed-directed bootstrap:** project the [monolithic seed's](MONOLITHIC_SEED_AND_SELF_GROUNDS.md) module-Ground graph to the specific interpreter and observation capabilities needed by the next intention, making acquisition lazy and evidence-qualified.

The fundamental goal remains: **the observer describes the information it needs; VL-64 determines which information to inspect, constructs a minimal sufficient Ground, and navigates that Ground without inventing unobserved structure.**
