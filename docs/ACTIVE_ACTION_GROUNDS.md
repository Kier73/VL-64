# Active Action-Conditioned Grounds

**Status:** implemented research experiment, November 18, 2026.  
**Core size:** no changes to `src/vld/`; budget remains at **65,093 / 65,536 bytes**.  
**Implementation:** [active_ground.py](../experiments/byte_fabric/active_ground.py), [active_adapter.py](../experiments/byte_fabric/active_adapter.py), [active_lab.py](../experiments/byte_fabric/active_lab.py).

## What changed

Previously, the Virtual Layer could fit an observed transition history and navigate a learned closed cycle or mathematical recurrence. It could also construct a smaller observation-preserving quotient of a **known finite deterministic model**.

The active experiment introduces a new question:

> **Given several Ground descriptions that agree with everything observed so far, which permitted experiment should the observer perform next to distinguish them?**

The observer supplies an **intention** (distinguish candidate descriptions), the host advertises an explicitly permitted action alphabet and reset mechanism, and the Virtual Layer chooses a bounded action sequence with the best predicted information gain per declared cost.

That closes a practical loop:

~~~text
competing candidate Grounds
       |
       v
predict outputs of candidate action words
       |
       v
choose most informative affordable word
       |
       v
qualified, resettable host actually observes resulting bytes
       |
       v
bind action + outputs + candidate elimination to evidence digest
       |
       v
refine competing Grounds or preserve contradiction
       |
       v
navigate retained action model (no remote action performed)
~~~

A host's installed probe callback is an **authority-bearing operation**. A Ground JSON file or an external observer request cannot register Python callbacks or grant itself permission to mutate a machine.

## Model family and scope

This prototype uses complete deterministic finite action machines, where each state/action pair has a successor and an **exact byte output**. It is a bounded passive/active *version-space learner*, not a general-purpose automatic reverse engineer.

- Candidate models have at most 128 states, 8 qualified actions and 256 output bytes per transition.
- A session admits 2–32 distinct candidate descriptions with the same action alphabet.
- Each candidate must contain a complete, structurally valid transition table; a content digest checks its internal integrity.
- A probe word is at most six actions. A bounded search examines up to 2,048 candidate words. Exhaustiveness relative to the declared search budget is reported separately.
- The model candidates are **proposals**, not observations, and a digest is **not a signature or a proof of correctness**.
- A resettable oracle is required so that competing predictions refer to the same initial state. A host declaration of reset behavior is not remote attestation.
- Read-only remote probes and controlled local simulated probes are distinct host policies. Arbitrary POST/PUT requests, shell execution, process mutation and model inference are not admitted.

## How the next experiment is selected

For candidate hypotheses \(M_1,\ldots,M_n\), VL-64 predicts each action word \(w\)'s exact byte-output trace:

\[
\operatorname{out}(M_i,w).
\]

Candidates with equal predicted traces are grouped into the same outcome partition. Under a declared **uniform prior**, expected information gain is

\[
\operatorname{IG}(w)=\log_2(n)-
\sum_{o}{|C_o|\over n}\log_2(|C_o|),
\]

where \(C_o\) is the subset of candidates predicting output trace \(o\). The selected word maximizes \(\operatorname{IG}(w)/\operatorname{declaredCost}(w)\), with deterministic tie-breaking. If the bounded search finds no informative word, the outcome is **UNRESOLVED**, not a statement that the candidates are equivalent.

This cost is a caller-supplied positive integer model, not calibrated wall time, network traffic or API charges. More powerful cost profiles and nonuniform priors remain research directions.

## Evidence-aware refinement

An actual probe returns one exact output byte sequence for every action of the chosen word. The observer records:

- its planned action word and the predictions made by every candidate;
- its expected information gain and declared cost;
- the source and host-declared authority type;
- the observed output bytes and SHA-256 digests;
- the before/after candidate Ground digests and eliminated candidates;
- a digest of the whole observation receipt.

A plan made for a prior version space cannot silently update a later one. The implementation cross-checks the proposed outcome partition and cost against the **current** registered candidates before accepting observed bytes.

If zero candidates match a real observation, the outcome is `CONTRADICTED_ALL_HYPOTHESES`. No candidate is fabricated to explain the counterexample. The contradiction is retained as a new evidence object. A future learner can create a **new** Ground with a different digest and an explicit derivation; it must not rewrite historical evidence.

If one candidate remains, it is **consistent with the executed experiments**, not proved equivalent to an unknown remote system across all action sequences.

## Internally navigable actions

A surviving candidate can be addressed with the same observer's typed request to navigate a repeated **modeled** action.

For a deterministic finite transition \(T_a\) induced by action \(a\),

\[
T_a^n = \underbrace{T_a\circ T_a\circ\cdots\circ T_a}_{n\text{ times}}.
\]

Instead of running the action \(n\) times, `ActionGroundNavigator.repeat` composes the finite successor transformation by repeated squaring, producing the **modeled terminal state and last action output** in \(\mathcal{O}(|S|\log n)\) work. It does not materialize the intervening output stream or send those actions to an external server.

The navigator explicitly labels the result `MODELED_ACTION_NAVIGATION`. The model's prediction remains separate from an independently executed remote result.

## Execute the experiment

From the repository root:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python -m experiments.byte_fabric.active_lab --out artifacts

python -m experiments.observer_intent.arbitrate \
  --intent artifacts/active-next-probe-intent.json \
  --out artifacts/recommended-experiment.json

python -m experiments.observer_intent.arbitrate \
  --intent artifacts/active-action-navigation-intent.json \
  --out artifacts/modeled-action-result.json
~~~

The first observer intention (`ground.active.plan`) recommends the next informative word from a saved catalogue. It **does not execute** the experiment. A host-side `QualifiedResetProbe` is what actually acquires byte observations. The second intention (`ground.action.navigate`) computes a modeled horizon without any external actuation.

### Controlled machine results

Four candidate three-state machines agree on all initial one-action observations. Only action sequences such as `advance → inspect` and `switch → inspect` reveal the hidden differences. VL-64 must discover these sequences, act against an **independent resettable reference implementation**, and re-plan after the first output.

The goal is to identify the one candidate consistent with all observations in **two discriminating experiments** (four action outputs), rather than exhausting arbitrary state/action traces. It then computes a one-trillion-step modeled action horizon using finite transformation powers.

There is no claim that this example is an unknown physical processor, a real remote state machine or an unsolved inference problem. The experiment exercises the architecture's ability to **choose what to observe** and preserve meaningful evidence.

### Real Internet byte experiment

~~~bash
python -m experiments.byte_fabric.active_lab \
  --out artifacts --live-only
~~~

The separate live branch uses **read-only HTTPS observations of public, pinned VL-64 GitHub files**. Each permitted action names a preapproved resource; the oracle returns SHA-256 bytes of the completely acquired response. Two different resource-identity candidates predict different byte digests, and the active planner chooses the resource that distinguishes them.

This establishes that the same typed action-and-evidence mechanism can operate on actual remote bytes under a restricted host policy. It does **not** establish that independent HTTP GETs expose a remotely evolving state machine. Provider availability or content changes may produce an explicit unresolved/contradictory result rather than a false certificate.

GitHub Actions runs both experiments and stores candidate Grounds, planned actions, selected model, observation receipts and result artifacts for independent inspection.

## Research references

- [A research agenda for active automata learning (2026)](https://sfortz.github.io/publication/2026-al-agenda/2026-al-agenda.pdf): observation tables, discrimination trees, and query selection.
- [An Abstract Framework for Counterexample Analysis in Active Automata Learning](https://proceedings.mlr.press/v34/isberner14a.pdf): refining learned hypotheses from counterexamples without claiming an unobserved equivalence.
- [Automata Learning with an Incomplete Teacher](https://www.cs.cornell.edu/~jnfoster/papers/lstar-with-blanks.pdf): incomplete observations and principled unknown outcomes.
- Existing cross-repository [finite transition-congruence work](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/observation-equivalence/transition_congruence.py) and [distinguishing-state navigation](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/observation-equivalence/distinguishing_navigation.py) informed the separation between evidence, hypotheses and state distinctions.

## Future frontier

1. **Generate hypotheses from real trace evidence:** the current candidates must be explicitly installed by a host. Build a bounded Mealy-learning algorithm with observation-table closure/consistency and independent held-out action words. Do not label this prototype as `L*`.
2. **Discriminate observational equivalence:** compare competing states under action suffixes; refine the state quotient with action-conditioned outputs. Reuse the compact `myhill_nerode_quotient` for completely known finite models, and preserve `UNKNOWN` for missing transitions.
3. **Learn meaningful host costs:** include request bytes, retries, wall time, rate limits, expected information gain, verification and recovery. Optimize total measured work, not just number of calls.
4. **Interactive remote observer-host contracts:** apply only operations that the host actually authorizes, with reset/replay, transport provenance and explicitly declared side effects. Differentiate idempotent reads from state-mutating actions.
5. **Parallel and federated observers:** allow another host to execute the same qualified experiment, compare their observations and challenge the surviving Ground without losing independent provenance.
6. **General typed interpretation:** use the byte fabric to discover action-specific variables, encoding and state transitions in documents, models, packet streams, screen buffers, and other byte-bearing sources, validating each abstraction before navigation.

The first step is deliberately **small enough to prove useful by execution** while the core remains within 64 KiB. The broader goal is a Virtual Layer that asks *which observation is most informative*, then navigates and compresses the resulting verified state description.
