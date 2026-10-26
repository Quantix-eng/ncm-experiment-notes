# What the record supports

[Experiment overview](../README.md#all-20-experiments-at-a-glance) · [Connections](../experiments/README.md) · [Related work](related-work.md) · [Reproduction](reproducing.md)

This is an exploratory record, assembled from campaigns completed in October 2026. Results were not generated again just to populate this repository.

## Observations and interpretations

A constant code is an observation about particular weights and inputs. The raw pooling identity is a separate mathematical statement. Neither establishes that pooling causes intermediate reversals.

A decoded intermediate value is an observation made through the decoder. It is not necessarily a decision fed back to the generator. In LM1B, agreement with the eventual final token is not a correctness measure.

An adapted continuation can help without showing a response specific to the donor. Recovery, crossed interaction and donor matching are different contrasts. Their definitions and control policies matter.

The terminal checker is external. Its conditional non-degradation argument assumes unique solutions, sound clue-respecting verification and exact fallback. It does not establish a learned verification mechanism or a whole-model speed benefit.

## Statistical scope

Cells, edit requests and multiple donors from one source are correlated. The retained studies generally group them by source puzzle or family. Shared donor banks, repeated templates, fixed model weights and reused evaluation sets still limit inference.

The 0.01-nat IOI margin is a conventional minimum effect, not a p-value threshold. An interval can exclude zero without establishing an effect larger than that margin. Report the continuous effect before attaching a pass/fail label.

Local freezes limit some choices within a campaign. They do not turn the whole exploratory history into a prospective preregistered study. Results on reused test material remain exploratory. A file hash proves consistency of a file, not that every historical choice was recorded.

## AI assistance and responsibility

AI tools contributed substantially to experiment planning, implementation, execution, analysis, audits and writing. Human direction and editorial decisions do not constitute an independent implementation of the experiments. Automated audits and same-workflow reruns are internal checks.

The repository does not claim an external review, an accepted paper or independent human replication. Anyone using the results should inspect the code and the assumptions, including the negative outcomes.

## Source provenance

Scientific scripts now live under each experiment’s `src/` or in `common/`. Local entry points provide isolated execution. The earlier archive remains in Git history; its reports are not duplicated in the working tree.

The [source inventory](../provenance/source-inventory.json) records hashes, relocation and omitted numerical arrays. Packaging checks and small examples added for this repository are recorded separately. They do not retroactively validate the scientific conclusions.
