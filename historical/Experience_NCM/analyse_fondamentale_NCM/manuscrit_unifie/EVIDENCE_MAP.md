# Carte des affirmations et de leurs preuves

Les identifiants E01-E14 sont utilisés dans le manuscrit. Les valeurs numériques sont reliées séparément à leurs pointeurs JSON dans `evidence/numbers.json`. Les noms de campagnes sont résolus dans `evidence/campaigns.json`. Aucun résultat n'est recalculé par cette carte.

## E01 - Equal within-window token histograms imply identical mean pooled additive token-plus-position embeddings.

Statut : Elementary algebra plus recorded operator audit. Section : `sec-pooling`.

Résultats : [mechanistic/pooling_algebra.json](evidence/records/mechanistic/pooling_algebra.json).
Code : [models/ncm_mac/_vendor/models/vqvae.py](code/models/ncm_mac/_vendor/models/vqvae.py), [history/mechanistic/scripts/pooling_algebra.py](code/history/mechanistic/scripts/pooling_algebra.py).
Traces compactes : [base_hard](evidence/traces/base_hard.npz).

Limite : The identity precedes GroupNorm and residual history; it does not force identical native quantizer codes.

## E02 - The hard level-5 code is constant on the examined clean data; replacing its prediction by code 11 while retaining the residual preserves recorded outputs.

Statut : Checkpoint observation and intervention. Section : `sec-pooling`.

Résultats : [mechanistic/level5_diagnostic.json](evidence/records/mechanistic/level5_diagnostic.json), [mechanistic/level5_interventions.json](evidence/records/mechanistic/level5_interventions.json), [generalization/sudoku_test.json](evidence/records/generalization/sudoku_test.json).
Code : [history/mechanistic/scripts/runtime.py](code/history/mechanistic/scripts/runtime.py), [history/mechanistic/scripts/level5_interventions.py](code/history/mechanistic/scripts/level5_interventions.py), [history/generalization/scripts/sudoku_levels.py](code/history/generalization/scripts/sudoku_levels.py).
Traces compactes : [base_hard](evidence/traces/base_hard.npz), [level5](evidence/traces/level5.npz).

Limite : Constancy is empirical. It is not zero activation, universal uselessness, an optimization-history explanation or a measured speedup.

## E03 - Position-weighted pooling breaks the raw averaging symmetry and retains recorded reconstruction while diversifying level-5 codes.

Statut : Operator construction and codec intervention. Section : `sec-repair`.

Résultats : [repair/heldout.json](evidence/records/repair/heldout.json), [repair/order_controls.json](evidence/records/repair/order_controls.json), [repair/parity.json](evidence/records/repair/parity.json), [repair/generator_alignment.json](evidence/records/repair/generator_alignment.json), [repair/noise.json](evidence/records/repair/noise.json).
Code : [history/repair/scripts/repair.py](code/history/repair/scripts/repair.py), [history/repair/scripts/experiments.py](code/history/repair/scripts/experiments.py).
Traces compactes : [repair_hard](evidence/traces/repair_hard.npz), [repair_external](evidence/traces/repair_external.npz), [repair_easy](evidence/traces/repair_easy.npz), [repair_medium](evidence/traces/repair_medium.npz).

Limite : Known solutions are encoded. The frozen native generator does not call the repaired encoding operator, and no downstream robustness gain is established.

## E04 - Decoded values corrected at 9 can become wrong again; all early native full-grid cutoffs fail on the recorded hard test.

Statut : Recorded trajectories and temporal interventions. Section : `sec-dynamics`.

Résultats : [temporal/observations.json](evidence/records/temporal/observations.json), [temporal/interventions.json](evidence/records/temporal/interventions.json), [temporal/reader.json](evidence/records/temporal/reader.json), [oracle/audit.json](evidence/records/oracle/audit.json).
Code : [history/temporal/scripts/observations.py](code/history/temporal/scripts/observations.py), [history/temporal/scripts/interventions.py](code/history/temporal/scripts/interventions.py), [history/temporal/scripts/runtime.py](code/history/temporal/scripts/runtime.py).
Traces compactes : [base_hard](evidence/traces/base_hard.npz), [temporal](evidence/traces/temporal.npz).

Limite : This falsifies strict decoded-value locking, not every possible latent commitment. Early cutoff does not isolate level 9 or establish a phase transition.

## E05 - Oracle encoding of correct Sudoku solutions produces correction-then-degradation trajectories without generator execution.

Statut : Subsystem sufficiency control. Section : `sec-dynamics`.

Résultats : [oracle/audit.json](evidence/records/oracle/audit.json), [oracle/verification.json](evidence/records/oracle/verification.json).
Code : [history/oracle/scripts/audit.py](code/history/oracle/scripts/audit.py), [models/ncm_mac/_vendor/models/quant.py](code/models/ncm_mac/_vendor/models/quant.py).
Traces compactes : [base_hard](evidence/traces/base_hard.npz), [base_easy](evidence/traces/base_easy.npz), [base_medium](evidence/traces/base_medium.npz).

Limite : Sufficiency is not exclusive causal attribution. Similar rates are not an attributable fraction or a statistical equivalence result.

## E06 - LM1B and TinyStories audits do not reproduce a constant native level, although a raw additive pooling symmetry remains.

Statut : Text tokenizer boundary cases. Section : `sec-pooling`.

Résultats : [generalization/lm1b_audit.json](evidence/records/generalization/lm1b_audit.json), [generalization/tinystories_audit.json](evidence/records/generalization/tinystories_audit.json), [generalization/lm1b_controls.json](evidence/records/generalization/lm1b_controls.json), [generalization/tinystories_controls.json](evidence/records/generalization/tinystories_controls.json).
Code : [history/generalization/scripts/text_audit.py](code/history/generalization/scripts/text_audit.py).
Traces compactes : [lm1b_codec](evidence/traces/lm1b_codec.npz), [tinystories_codec](evidence/traces/tinystories_codec.npz).

Limite : Two tokenizers, finite text samples and different codebook capacities do not identify an architecture-wide defect or demonstrate every level useful.

## E07 - Raw final-token adoption followed by abandonment occurs in LM1B generation; true reconstruction corrections also revert in the oracle codec.

Statut : Text generation and oracle-codec trajectories. Section : `sec-dynamics`.

Résultats : [crossed/lm1b_generation.json](evidence/records/crossed/lm1b_generation.json), [crossed/lm1b_oracle.json](evidence/records/crossed/lm1b_oracle.json), [crossed/lm1b_asset.json](evidence/records/crossed/lm1b_asset.json), [crossed/lm1b_checkpoint_structure.json](evidence/records/crossed/lm1b_checkpoint_structure.json).
Code : [history/crossed/scripts/text_generation.py](code/history/crossed/scripts/text_generation.py), [history/crossed/scripts/assets.py](code/history/crossed/scripts/assets.py), [history/crossed/scripts/transfer.py](code/history/crossed/scripts/transfer.py).
Traces compactes : [lm1b_generation](evidence/traces/lm1b_generation.npz), [lm1b_oracle](evidence/traces/lm1b_oracle.npz).

Limite : The final generated token is a retrospective anchor, not ground truth. Raw and heuristic readouts are distinct. Invalid-loader trajectories are excluded, not results.

## E08 - The joint recovery-and-specificity criterion for omitting residual 9 is not met on the new 256-puzzle corpus.

Statut : Four-condition intervention and announced contrasts. Section : `sec-crossed`.

Résultats : [crossed/fresh_crossed.json](evidence/records/crossed/fresh_crossed.json), [crossed/hard_test_crossed.json](evidence/records/crossed/hard_test_crossed.json), [crossed/external_previous_crossed.json](evidence/records/crossed/external_previous_crossed.json), [crossed/protocol_frozen.json](evidence/records/crossed/protocol_frozen.json).
Code : [history/crossed/scripts/crossed.py](code/history/crossed/scripts/crossed.py), [history/crossed/scripts/runtime.py](code/history/crossed/scripts/runtime.py).
Traces compactes : [crossed_fresh](evidence/traces/crossed_fresh.npz), [crossed_test](evidence/traces/crossed_test.npz), [crossed_external](evidence/traces/crossed_external.npz).

Limite : One omission, checkpoint and dose. No proof of globally blind adaptation, no equivalence claim and no exclusion of all forms of compensation.

## E09 - Frozen target-code substitutions write requested different digits in 784/1024 trials versus 0/1024 Gaussian controls on 128 new sources.

Statut : Terminal code substitution. Section : `sec-terminal`.

Résultats : [terminal/writing_interventions.json](evidence/records/terminal/writing_interventions.json), [terminal/numerical_support_audit.json](evidence/records/terminal/numerical_support_audit.json), [terminal/pilot_format_audit.json](evidence/records/terminal/pilot_format_audit.json), [terminal/discovery_tables.json](evidence/records/terminal/discovery_tables.json).
Code : [history/terminal/scripts/experiment.py](code/history/terminal/scripts/experiment.py), [history/terminal/scripts/verify.py](code/history/terminal/scripts/verify.py).
Traces compactes : [terminal_writing](evidence/traces/terminal_writing.npz), [terminal_pilot](evidence/traces/terminal_pilot.npz).

Limite : Control matches local norm/support, not codebook geometry. No proof of new semantics, multi-cell compositionality or total context independence.

## E10 - Lookup supplies externally certified grids for 552/1024 new hard puzzles; checkpoint-specific secondary coverage is also recorded.

Statut : Recorded lookup coverage and quality. Section : `sec-terminal`.

Résultats : [terminal/hard_readout.json](evidence/records/terminal/hard_readout.json), [terminal/easy_readout.json](evidence/records/terminal/easy_readout.json), [terminal/medium_readout.json](evidence/records/terminal/medium_readout.json), [terminal/primary_decision.json](evidence/records/terminal/primary_decision.json), [terminal/protocol_frozen.json](evidence/records/terminal/protocol_frozen.json).
Code : [history/terminal/scripts/policy.py](code/history/terminal/scripts/policy.py), [history/terminal/scripts/base.py](code/history/terminal/scripts/base.py), [history/terminal/scripts/experiment.py](code/history/terminal/scripts/experiment.py).
Traces compactes : [terminal_hard](evidence/traces/terminal_hard.npz), [terminal_easy](evidence/traces/terminal_easy.npz), [terminal_medium](evidence/traces/terminal_medium.npz).

Limite : The 30% coverage threshold followed favorable development results. Coverage does not establish novelty or an end-to-end performance gain.

## E11 - A sound validity-and-clue checker plus exact fallback cannot reduce exact success on unique-solution puzzles.

Statut : Conditional mathematical guarantee. Section : `sec-terminal`.

Résultats : [terminal/verification.json](evidence/records/terminal/verification.json), [terminal/editorial_revision_audit.json](evidence/records/terminal/editorial_revision_audit.json).
Code : [history/terminal/scripts/policy.py](code/history/terminal/scripts/policy.py), [history/terminal/scripts/base.py](code/history/terminal/scripts/base.py), [history/terminal/scripts/verify.py](code/history/terminal/scripts/verify.py).
Traces compactes : [terminal_hard](evidence/traces/terminal_hard.npz).

Limite : A general verification argument, not a learned NCM mechanism. Multiple-solution reference fidelity and correctness of rejected fallback outputs are not guaranteed.

## E12 - Readout arithmetic means differ in the recorded session; the announced full-pipeline geometric interval includes one.

Statut : Historical paired CPU timing. Section : `sec-terminal`.

Résultats : [terminal/benchmark.json](evidence/records/terminal/benchmark.json), [terminal/benchmark_raw.json](evidence/records/terminal/benchmark_raw.json).
Code : [history/terminal/scripts/benchmark.py](code/history/terminal/scripts/benchmark.py), [history/terminal/scripts/policy.py](code/history/terminal/scripts/policy.py).
Traces compactes : Pas de trace compacte supplémentaire ; mesures et/ou preuve formelle référencées..

Limite : One CPU session, 64 sources, all generator levels retained. No established end-to-end speedup or GPU/energy claim.

## E13 - The synthesis preserves eight campaign histories, dependencies, local protocols and their recorded verification scopes.

Statut : Design and provenance records. Section : `sec-validity`.

Résultats : [terminal/editorial_revision_baseline.json](evidence/records/terminal/editorial_revision_baseline.json), [terminal/editorial_revision_audit.json](evidence/records/terminal/editorial_revision_audit.json), [crossed/input_manifest.json](evidence/records/crossed/input_manifest.json), [base/input_manifest.json](evidence/records/base/input_manifest.json).
Code : [history/base/scripts/common.py](code/history/base/scripts/common.py), [history/base/scripts/upstream_parity.py](code/history/base/scripts/upstream_parity.py), [history/crossed/scripts/verify.py](code/history/crossed/scripts/verify.py).
Traces compactes : [base_hard](evidence/traces/base_hard.npz).

Limite : Local freezes and same-agent software assertions are not external preregistration, independent review or proof against selection bias.

## E14 - Initial confidence and conditional-revision probe results do not establish added hidden-state value or native metacognitive control.

Statut : Other recorded probe and intervention outcomes. Section : `app-secondary`.

Résultats : [base/probes.json](evidence/records/base/probes.json), [base/causal.json](evidence/records/base/causal.json), [mechanistic/conditional_probes.json](evidence/records/mechanistic/conditional_probes.json), [mechanistic/conditional_external.json](evidence/records/mechanistic/conditional_external.json), [mechanistic/steering_conditional.json](evidence/records/mechanistic/steering_conditional.json).
Code : [history/base/scripts/analyze.py](code/history/base/scripts/analyze.py), [history/base/scripts/experiments.py](code/history/base/scripts/experiments.py), [history/mechanistic/scripts/conditional.py](code/history/mechanistic/scripts/conditional.py).
Traces compactes : Pas de trace compacte supplémentaire ; mesures et/ou preuve formelle référencées..

Limite : Probe families, fitting splits and observed baselines limit inference; these outcomes are retained rather than silently replaced by the four-pillar narrative.
