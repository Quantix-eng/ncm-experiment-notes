"""Audit the v0.2 standardization failure without changing or refitting probes."""
import json

import numpy as np

from run_pilot import ROOT, file_hash, write_json


def meaning(column):
    if column < 1341:
        return dict(family="puzzle_or_summary", column=column)
    if column < 2409:
        position, token = divmod(column - 1341, 12)
        return dict(family="current_decoder_probability", position=position, token=token,
                    expected_row_separator=position % 10 == 9, zero_based=True)
    if column < 4265:
        return dict(family="code_probability", column=column)
    return dict(family="decoded_history_one_hot", column=column)


def main():
    directory = ROOT / "audit/metacognition/development-primary-v0.2"
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "complete"
    cache = ROOT / "audit/metacognition/train-features-v0.2"
    x = np.load(cache / "rich.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(cache / "indices.npy", allow_pickle=False)
    records = []
    for fold in range(5):
        path = directory / f"outer-{fold}-rich.npz"
        assert file_hash(path) == report["artifacts"][path.name]["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            validation = np.isin(indices, saved["validation_indices"])
            train, valid = np.asarray(x[~validation]), np.asarray(x[validation])
            kept = saved["base_keep"]
            z = (valid[:, kept] - saved["base_mean"][kept]) / saved["base_scale"]
            row, coordinate = np.unravel_index(np.argmax(np.abs(z)), z.shape)
            column = int(np.flatnonzero(kept)[coordinate])
            contribution = z * saved["weight"]
            records.append(dict(fold=fold, retained_features=int(kept.sum()),
                retained_exactly_constant_columns=int(np.sum(np.ptp(train[:, kept], axis=0) == 0)),
                retained_columns_std_below_1e_8=int((saved["base_scale"] < 1e-8).sum()),
                largest_abs_standardized_validation_value=float(np.max(np.abs(z))),
                largest_abs_single_feature_logit_contribution=float(np.max(np.abs(contribution))),
                largest_abs_validation_logit=float(np.max(np.abs(saved["validation_logits"]))),
                most_amplified_feature=dict(column=column, meaning=meaning(column),
                    training_min=float(train[:, column].min()), training_max=float(train[:, column].max()),
                    training_std=float(saved["base_scale"][coordinate]),
                    validation_value=float(valid[row, column]), sample_index=int(indices[validation][row]))))
    output = dict(status="diagnosed_no_refit", report_sha256=file_hash(directory / "report.json"), folds=records,
        conclusion="Nonzero but tiny training variances amplify rare validation probabilities. This is a failure mode of the specified preprocessing, not evidence of altered NCM outputs or train/test leakage.",
        action="Keep v0.2 results; define a versioned development-only preprocessing amendment before any new fit. Do not interpret v0.2 AUC differences as robust evidence for or against internal information.")
    write_json(directory / "scaling-diagnostic.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
