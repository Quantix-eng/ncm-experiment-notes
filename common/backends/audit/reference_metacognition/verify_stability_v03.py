"""Replay the twenty bootstrap predictors and check resampled group boundaries."""
import json

import numpy as np

from final_v03 import transformed
from probes import metrics
from run_pilot import ROOT, file_hash, write_json
from study_v03 import load_training


def main():
    directory = ROOT / "audit/metacognition/stability-v0.3"
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "complete"
    _, _, data, _, indices, y, fold, _ = load_training()
    validation = np.flatnonzero(fold == 0)
    expected = indices[validation]
    selected = {k: np.asarray(v[validation]) for k, v in data.items()}
    error = 0.
    rng = np.random.default_rng(20261028)
    pool = np.flatnonzero(fold != 0)
    for repeat in range(20):
        sampled = np.concatenate([rng.choice(pool[y[pool] == c], len(pool[y[pool] == c]), replace=True) for c in (0, 1)])
        for model in report["models"]:
            path = directory / f"outer-{repeat}-{model}.npz"
            assert file_hash(path) == report["artifacts"][path.name]["sha256"]
            with np.load(path, allow_pickle=False) as saved:
                assert np.array_equal(saved["fitting_indices"], indices[sampled])
                assert np.array_equal(saved["validation_indices"], expected)
                assert not set(saved["fitting_indices"]) & set(expected)
                replay = transformed(saved, selected, model) @ saved["weight"] + saved["intercept"]
                error = max(error, float(np.max(np.abs(replay - saved["validation_logits"]))))
                assert np.allclose(replay, saved["validation_logits"], rtol=5e-13, atol=1e-10)
                assert metrics(y[validation], saved["validation_logits"]) == report["completed_outer_folds"][repeat]["selection"][model]["validation_metrics"]
    output = dict(status="passed", report_sha256=file_hash(directory / "report.json"), repeats=20,
                  max_abs_replay_error=error, resampling_indices_reconstructed=True, no_fitting_validation_group_overlap=True)
    write_json(directory / "verification.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
