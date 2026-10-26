"""Independent arithmetic replay of the locked held-out analysis, no fitting."""
import json

import numpy as np

from probes import auc, metrics
from run_pilot import ROOT, file_hash, write_json


def replay(saved, features, name):
    family = "summary" if name == "summary" else "rich"
    keep = saved["base_keep"]
    x = (features[family][:, keep] - saved["base_mean"][keep]) / saved["base_scale"]
    assert (saved["base_scale"] >= .1).all()
    if name == "rich_hidden_primary":
        keep = saved["hidden_keep"]
        z = (features["hidden_raw"][:, keep] - saved["hidden_mean"][keep]) / saved["hidden_scale"]
        z = (z @ saved["pca_axes"].T - saved["pca_score_mean"]) / saved["pca_score_scale"]
        x = np.concatenate([x, z], axis=1)
    elif name == "rich_noise_control":
        noise = np.random.default_rng(20261024).standard_normal((2000, 16))[features["indices"]]
        keep = saved["noise_keep"]
        z = (noise[:, keep] - saved["noise_mean"][keep]) / saved["noise_scale"]
        x = np.concatenate([x, z], axis=1)
    return x @ saved["weight"] + saved["intercept"]


def main():
    directory = ROOT / "audit/metacognition/final-v0.3"
    lock = json.loads((directory / "lock.json").read_text())
    report = json.loads((directory / "analysis/report.json").read_text())
    cal = json.loads((directory / "analysis/calibration.json").read_text())
    extraction = json.loads((directory / "held-out/report.json").read_text())
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    assert report["status"] == extraction["status"] == "complete"
    assert report["lock_sha256"] == cal["lock_sha256"] == extraction["lock_sha256"] == file_hash(directory / "lock.json")
    for name, digest in lock["source_files"].items():
        assert file_hash(ROOT / name) == digest
    for name, digest in lock["model_files"].items():
        assert file_hash(directory / name) == digest
    for name, entry in extraction["artifacts"].items():
        assert file_hash(directory / "held-out" / name) == entry["sha256"]
    for name, digest in report["artifacts"].items():
        assert file_hash(directory / "analysis" / name) == digest
    with np.load(directory / "analysis/test-predictions.npz", allow_pickle=False) as saved_scores:
        for partition in ("calibration", "test"):
            with np.load(directory / f"held-out/{partition}-features.npz", allow_pickle=False) as features:
                assert features["indices"].tolist() == split["partition_indices"][partition]
                assert int(features["failure"].sum()) == split["class_counts"][partition]["failures"]
                assert not set(features["indices"]) & set(split["partition_indices"]["train"])
                y = features["failure"]
                for model in lock["models"]:
                    with np.load(directory / f"model-{model}.npz", allow_pickle=False) as fitted:
                        logits = replay(fitted, features, model)
                    adjusted = logits * cal["platt"][model]["slope"] + cal["platt"][model]["intercept"]
                    expected = report["results"][model] if partition == "test" else cal["metrics"][model]
                    for kind, score in (("raw", logits), ("calibrated", adjusted)):
                        for key, value in metrics(y, score).items():
                            assert np.isclose(value, expected[kind][key], rtol=1e-10, atol=1e-12)
                    if partition == "test":
                        assert np.allclose(logits, saved_scores[model], rtol=1e-12, atol=1e-10)
        y = saved_scores["failure"]
        rng = np.random.default_rng(lock["bootstrap"]["seed"])
        p, n = np.flatnonzero(y), np.flatnonzero(y == 0)
        delta = []
        for _ in range(5000):
            rows = np.r_[rng.choice(p, len(p), replace=True), rng.choice(n, len(n), replace=True)]
            delta.append(auc(y[rows], saved_scores["rich_hidden_primary"][rows]) - auc(y[rows], saved_scores["rich"][rows]))
        assert np.array_equal(np.quantile(delta, [.025, .975]), report["primary_paired_bootstrap_95"])
        assert auc(y, saved_scores["rich_hidden_primary"]) - auc(y, saved_scores["rich"]) == report["primary_delta_auc"]
    write_json(directory / "analysis/verification.json", dict(status="passed", report_sha256=file_hash(directory / "analysis/report.json"),
        checks=["lock/source/model/artifact hashes", "reserved partition indices", "independent transform replay", "raw and calibrated metrics", "5000 paired bootstrap resamples"],
        model_execution=False, probe_fitting=False))
    print("Final locked analysis verified without refitting")


if __name__ == "__main__":
    main()
