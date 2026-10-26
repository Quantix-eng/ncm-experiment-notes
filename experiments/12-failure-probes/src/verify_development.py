"""Replay saved outer-fold predictors without refitting or loading the NCM."""
import argparse
import json
from pathlib import Path

import numpy as np

from probes import auc, metrics
from run_pilot import ROOT, file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "audit/metacognition/development-primary-v0.2")
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    assert report["status"] == "complete"
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    cache = ROOT / "audit/metacognition/train-features-v0.2"
    manifest = json.loads((cache / "manifest.json").read_text())
    for name, fingerprint in manifest["files"].items():
        assert file_hash(cache / name) == fingerprint
    for name, record in report["artifacts"].items():
        assert file_hash(args.directory / name) == record["sha256"]
    indices = np.load(cache / "indices.npy", allow_pickle=False)
    y = np.load(cache / "failure.npy", allow_pickle=False)
    assert indices.tolist() == split["partition_indices"]["train"]
    assert not set(indices) & (set(split["partition_indices"]["test"]) | set(split["partition_indices"]["calibration"]))
    data = {k: np.load(cache / f"{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in ("rich", "puzzle", "summary", "hidden_raw")}
    noise = np.random.default_rng(20261024).standard_normal((2000, 16))[indices]
    maximum_error = 0.
    with np.load(args.directory / "oof-progress.npz", allow_pickle=False) as oof:
        assert np.array_equal(oof["indices"], indices) and np.array_equal(oof["failure"], y)
        expected_fold = np.array([split["train_cv_fold"][str(i)] for i in indices])
        assert np.array_equal(oof["fold"], expected_fold)
        for model in report["models"]:
            for fold in range(5):
                rows = np.flatnonzero(expected_fold == fold)
                with np.load(args.directory / f"outer-{fold}-{model}.npz", allow_pickle=False) as saved:
                    assert np.array_equal(saved["validation_indices"], indices[rows])
                    base = data[model if model in ("puzzle", "summary") else "rich"][rows]
                    x = (base[:, saved["base_keep"]] - saved["base_mean"][saved["base_keep"]]) / saved["base_scale"]
                    if model == "rich_noise_control":
                        extra = (noise[rows][:, saved["noise_keep"]] - saved["noise_mean"][saved["noise_keep"]]) / saved["noise_scale"]
                        x = np.column_stack([x, extra])
                    elif model in ("rich_hidden_primary", "hidden_shuffle_control"):
                        source = saved["hidden_validation_source_rows"]
                        assert sorted(source.tolist()) == rows.tolist()
                        if model == "rich_hidden_primary":
                            assert np.array_equal(source, rows)
                        extra = data["hidden_raw"][source]
                        extra = (extra[:, saved["hidden_keep"]] - saved["hidden_mean"][saved["hidden_keep"]]) / saved["hidden_scale"]
                        extra = (extra @ saved["pca_axes"].T - saved["pca_score_mean"]) / saved["pca_score_scale"]
                        x = np.column_stack([x, extra])
                    replay = x @ saved["weight"] + saved["intercept"]
                    error = float(np.max(np.abs(replay - oof[model][rows])))
                    maximum_error = max(maximum_error, error)
                    # Matrix strides can change FP64 BLAS reduction order; some
                    # v0.2 logits exceed 1e6 because of ill-scaled rare features.
                    assert np.allclose(replay, oof[model][rows], rtol=5e-13, atol=1e-10), (model, fold, error)
                    assert np.array_equal(saved["validation_logits"], oof[model][rows])
            assert metrics(y, oof[model]) == report["results"][model]
        output = dict(status="passed", n=len(y), report_sha256=file_hash(args.directory / "report.json"),
                      max_abs_logit_replay_error=maximum_error, calibration_or_test_used=False,
                      note="Independent numerical replay of saved preprocessing and coefficients; not independent model replication or learning-variability inference")
        if "rich" in oof and "rich_hidden_primary" in oof:
            rng = np.random.default_rng(20261027)
            positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            deltas = []
            for _ in range(5000):
                selected = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
                deltas.append(auc(y[selected], oof["rich_hidden_primary"][selected]) - auc(y[selected], oof["rich"][selected]))
            output["descriptive_pooled_oof_delta_auc_percentile_95"] = np.quantile(deltas, [.025, .975]).tolist()
            output["bootstrap_caveat"] = "Fixed pooled OOF predictions from overlapping training sets; descriptive only, excludes refitting uncertainty, not the protocol's final-test interval"
    write_json(args.directory / "verification.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
