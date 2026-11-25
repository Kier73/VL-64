# Finite nonlinear observer closure — CLT-004 translation

Learn the least linear space of Boolean observations closed under pullback by declared nonlinear state maps. For observer o and actions F_a, repeatedly adjoin b∘F_a to a GF(2) basis of functions on a finite domain. This produces matrices M_a and an observer row d with B(F_a(x))=M_a B(x), o(x)=dB(x). Matrix composition uses the native VL-64 monoid_power; independent rank/closure checks use native ExactFieldMatrix and compare state classes with native myhill_nerode_quotient.

The separate experiment leaves src/vld entirely unchanged. It learns from complete action tables, not partial observations or pretrained states. An 8-bit source has four visible bits and four nuisance bits. Rotation alone: rank 4, 16 observer classes. Nonlinear shear: rank 12, still 16 classes. Add a hidden-to-visible coupling: rank 25, 31 classes. The feature rank can exceed original state bit width; it is not a claim of storage compression. Rank budget 16 refuses the coupled case. A separate JS mathematical control executed before this commit checked every transition and 768 long-word comparisons per family; native Python tests are CI-gated. The native tests additionally include the coupling action in long navigation, rank budget and tampered operator controls. No claim that powering beats source cycle navigation or lookup.

Run from repo root:

```sh
PYTHONPATH=src python -m unittest discover -s experiments/observer_closure -v
```

Prior public CLT-003: https://github.com/Kier73/In-Formation-Space/pull/382. Next: compare linear feature refinements with actual reachable-state nonlinear witnesses in VL-I; acquire source pullbacks selectively under independent evidence rather than full tables.
