# Related work and prior findings

[Experiment overview](../README.md#all-20-experiments-at-a-glance) · [Connections](../experiments/README.md) · [Scope and provenance](scope.md) · [Reproduction](reproducing.md)

- [Next-Concept Modeling](https://arxiv.org/abs/2610.14162): the model, checkpoints and original experiments. The intermediate translation example already acknowledges non-monotonic quality. Our measurements are additional checkpoint observations, not priority for that general phenomenon.
- [Interpretability in the Wild](https://arxiv.org/abs/2211.00593): the GPT-2 IOI circuit and its backup components. Recovery in this setting is not a new discovery here.
- [The Hydra Effect](https://arxiv.org/abs/2307.15771): downstream compensation after internal ablations.
- [Explorations of Self-Repair](https://arxiv.org/abs/2402.15390): resampling interventions, self-repair and final LayerNorm effects. Different losses and control policies need not measure the same thing.
- [Conditional Co-Ablation / CoAx](https://arxiv.org/abs/2607.01940): conditional importance of backup components and causal validation after primary removal. Its relation to our donor-matching contrast needs further work; we do not claim a stronger method.
- [MIB](https://arxiv.org/abs/2504.13151): circuit and causal-variable localization benchmarks. The IOI study uses public prompts, without being an official benchmark submission.
- [InterpBench](https://arxiv.org/abs/2407.14494): semi-synthetic networks and reference programs. The retained study found a fidelity problem on one tested case; that observation has not been established as a general flaw in the benchmark.

These links are starting points rather than an exhaustive literature review. Older campaign bibliographies remain in their original reports. No benchmark lead or novelty claim is inferred from the size of that bibliography.
