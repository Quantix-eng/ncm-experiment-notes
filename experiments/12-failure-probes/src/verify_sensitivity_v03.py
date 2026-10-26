"""Independent replay of the prespecified secondary level analyses."""
import json

import numpy as np

from probes import auc, metrics
from run_pilot import ROOT, file_hash, write_json
from verify_final_v03 import replay


def main():
    directory = ROOT / "audit/metacognition/sensitivity-v0.3"
    lock = json.loads((directory / "lock.json").read_text())
    report = json.loads((directory / "analysis/report.json").read_text())
    extraction = json.loads((directory / "held-out/report.json").read_text())
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    assert report["status"] == extraction["status"] == "complete"
    assert report["lock_sha256"] == extraction["lock_sha256"] == file_hash(directory / "lock.json")
    for name, digest in lock["source_files"].items():
        assert file_hash(ROOT / "experiments/metacognition" / name) == digest
    for name, digest in lock["models_sha256"].items():
        assert file_hash(directory / name) == digest
    for name, digest in extraction["artifacts"].items():
        assert file_hash(directory / "held-out" / name) == digest
    for name, digest in report["artifacts"].items():
        assert file_hash(directory / "analysis" / name) == digest
    for level in lock["levels"]:
        results = report["results"][str(level)]
        platt = json.loads((directory / f"analysis/calibration-{level}.json").read_text())
        with np.load(directory / f"analysis/test-predictions-{level}.npz", allow_pickle=False) as saved:
            with np.load(directory / f"held-out/test-{level}.npz", allow_pickle=False) as features:
                assert features["indices"].tolist() == split["partition_indices"]["test"]
                assert np.array_equal(features["failure"], saved["failure"])
                y = saved["failure"]
                for model in lock["models"]:
                    with np.load(directory / f"model-{level}-{model}.npz", allow_pickle=False) as fitted:
                        logits = replay(fitted, features, model)
                    assert np.allclose(logits, saved[model], rtol=1e-12, atol=1e-10)
                    adjusted = logits * platt[model]["slope"] + platt[model]["intercept"]
                    for kind, score in (("raw", logits), ("calibrated", adjusted)):
                        for key, value in metrics(y, score).items():
                            assert np.isclose(value, results["results"][model][kind][key], atol=1e-12, rtol=1e-10)
            rng = np.random.default_rng(20261027)
            p, n = np.flatnonzero(y), np.flatnonzero(y == 0)
            deltas = []
            for _ in range(5000):
                rows = np.r_[rng.choice(p, len(p), replace=True), rng.choice(n, len(n), replace=True)]
                deltas.append(auc(y[rows], saved["rich_hidden_primary"][rows]) - auc(y[rows], saved["rich"][rows]))
            assert np.array_equal(np.quantile(deltas, [.025, .975]), results["descriptive_bootstrap_95"])
            assert auc(y, saved["rich_hidden_primary"]) - auc(y, saved["rich"]) == results["delta_auc"]
    write_json(directory / "analysis/verification.json", dict(status="passed", report_sha256=file_hash(directory / "analysis/report.json"),
        levels=lock["levels"], independent_replay=True, bootstrap_recomputed=True, model_execution=False, fitting=False))
    print("Secondary analyses independently verified")


if __name__ == "__main__":
    main()
