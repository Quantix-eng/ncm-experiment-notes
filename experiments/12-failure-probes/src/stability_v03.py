"""Corrected group bootstrap; keep the original completed nested source frozen."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from probes import stratified_folds
from run_pilot import ROOT, file_hash, write_json
from study_v03 import HERE, load_training, select_and_fit


def inner_assignments(y, pool, sampled):
    mapping = np.full(len(y), -1, dtype=int)
    mapping[pool] = stratified_folds(y[pool], 20261029)
    cv = mapping[sampled]
    for k in range(5):
        assert (cv == k).any() and (cv != k).any()
        assert not set(sampled[cv == k]) & set(sampled[cv != k])
    return cv


def main():
    directory = ROOT / "audit/metacognition/stability-v0.3"
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    protocol, split, data, cache, indices, y, fold, noise = load_training()
    amendment = json.loads((HERE / "amendment-v0.3.json").read_text())
    models = amendment["stability"]["models"]
    report = dict(status="running", mode="stability", models=models, fits=[], completed_outer_folds=[],
        calibration_partition_used=False, test_partition_used=False,
        amendment_sha256=file_hash(HERE / "amendment-v0.3.json"),
        correction_sha256=file_hash(HERE / "stability-correction-v0.3.json"),
        source_files={name: file_hash(HERE / name) for name in ("stability_v03.py", "study_v03.py", "probes.py", "features.py", "train_probes.py")},
        feature_cache_manifest_sha256=file_hash(cache / "manifest.json"))
    start = time.perf_counter()
    write_json(directory / "report.json", report)
    try:
        rng = np.random.default_rng(20261028)
        pool, valid = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
        for k in range(20):
            train = np.concatenate([rng.choice(pool[y[pool] == c], len(pool[y[pool] == c]), replace=True) for c in (0, 1)])
            cv = inner_assignments(y, pool, train)
            fitted, diagnostics = select_and_fit(data, y, train, valid, cv, models, noise,
                20261025 + 100 * k, protocol, progress=f"stability {k + 1}/20")
            selection = {}
            for name, (fit, preprocessing, logits, diagnostic) in fitted.items():
                np.savez_compressed(directory / f"outer-{k}-{name}.npz", weight=fit["weight"], intercept=np.array(fit["intercept"]),
                    validation_indices=indices[valid], validation_logits=logits, fitting_indices=indices[train], **preprocessing)
                selection[name] = diagnostic
            report["fits"].extend(dict(repeat=k, **d) for d in diagnostics)
            report["completed_outer_folds"].append(dict(fold=k, n_train=len(train), n_unique_train=len(set(train)),
                                                      n_validation=len(valid), selection=selection))
            write_json(directory / "report.json", report)
        deltas = [r["selection"]["rich_hidden_primary"]["validation_metrics"]["auc"] - r["selection"]["rich"]["validation_metrics"]["auc"] for r in report["completed_outer_folds"]]
        report["delta_auc_sensitivity"] = dict(values=deltas, mean=float(np.mean(deltas)), median=float(np.median(deltas)),
            range=[min(deltas), max(deltas)], descriptive_percentiles_2_5_97_5=np.quantile(deltas, [.025, .975]).tolist(),
            caveat="Fixed development holdout fold 0, 20 fitting-group bootstraps; not an inferential CI or power estimate")
        for name, digest in report["source_files"].items():
            assert file_hash(HERE / name) == digest
        report.update(status="complete", wall_seconds=time.perf_counter() - start,
                      artifacts={p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in directory.glob("*.npz")})
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        write_json(directory / "report.json", report)
        raise
    write_json(directory / "report.json", report)
    print(json.dumps(dict(status=report["status"], sensitivity=report["delta_auc_sensitivity"])))


if __name__ == "__main__":
    main()
