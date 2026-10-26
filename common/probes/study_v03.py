"""Versioned, stable development of the original probes; v0.2 stays immutable."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch

from probes import HiddenPCA, Standardizer, auc, fit_logistic, log_loss, metrics, predict, stratified_folds
from run_pilot import ROOT, file_hash, write_json
from train_probes import feature_cache

HERE = Path(__file__).resolve().parent


class ObservableScaler(Standardizer):
    def fit(self, x):
        self.mean = x.mean(0)
        self.keep = np.ptp(x, axis=0) > 0
        self.scale = np.maximum(x.std(0, ddof=0)[self.keep], .1)
        return self


def prepared_pair(data, train, valid, models, noise, seed):
    output = {}
    for family in ("puzzle", "summary", "rich"):
        relevant = [m for m in models if (family == "rich" and m not in ("puzzle", "summary")) or m == family]
        if not relevant:
            continue
        scaler = ObservableScaler().fit(np.asarray(data[family][train]))
        a, b = scaler.transform(data[family][train]), scaler.transform(data[family][valid])
        base = dict(base_mean=scaler.mean, base_keep=scaler.keep, base_scale=scaler.scale)
        for name in relevant:
            if name in ("puzzle", "summary", "rich"):
                output[name] = (a, b, base)
            elif name == "rich_noise_control":
                ns = Standardizer().fit(noise[train])
                output[name] = (np.column_stack([a, ns.transform(noise[train])]),
                                np.column_stack([b, ns.transform(noise[valid])]),
                                dict(**base, noise_mean=ns.mean, noise_keep=ns.keep, noise_scale=ns.scale))
            else:
                ht, hv = np.asarray(data["hidden_raw"][train]), np.asarray(data["hidden_raw"][valid])
                source = valid.copy()
                if name == "hidden_shuffle_control":
                    rng = np.random.default_rng(seed)
                    ht = ht[rng.permutation(len(ht))]
                    order = rng.permutation(len(hv))
                    hv, source = hv[order], valid[order]
                pca = HiddenPCA().fit(ht)
                output[name] = (np.column_stack([a, pca.transform(ht)]), np.column_stack([b, pca.transform(hv)]),
                    dict(**base, hidden_mean=pca.standardizer.mean, hidden_keep=pca.standardizer.keep,
                         hidden_scale=pca.standardizer.scale, pca_axes=pca.axes,
                         pca_score_mean=pca.score_scaler.mean, pca_score_scale=pca.score_scaler.scale,
                         hidden_validation_source_rows=source))
    return output


def load_training():
    protocol_path = HERE / "protocol-v0.2.json"
    split_path = ROOT / "audit/metacognition/split-v0.2.json"
    protocol, split = json.loads(protocol_path.read_text()), json.loads(split_path.read_text())
    assert file_hash(protocol_path) == split["protocol_sha256"]
    data, cache = feature_cache(ROOT / "audit/metacognition/train-v0.2", protocol, split)
    indices, y = np.asarray(data["indices"]), np.asarray(data["failure"])
    assert indices.tolist() == split["partition_indices"]["train"]
    fold = np.array([split["train_cv_fold"][str(i)] for i in indices])
    noise = np.random.default_rng(20261024).standard_normal((2000, 16))[indices]
    return protocol, split, data, cache, indices, y, fold, noise


def select_and_fit(data, y, train, valid, cv, models, noise, seed, protocol, progress=None):
    strengths = protocol["fit"]["lambda_grid"]
    losses = {m: {str(s): [] for s in strengths} for m in models}
    diagnostics = []
    for k in range(5):
        it, iv = train[cv != k], train[cv == k]
        assert not set(it) & set(iv) and not set(train) & set(valid)
        prepared = prepared_pair(data, it, iv, models, noise, seed + k)
        for name, (a, b, _) in prepared.items():
            for strength in strengths:
                fit = fit_logistic(a, y[it], strength, protocol["fit"]["optimizer_settings"])
                loss = log_loss(y[iv], predict(fit, b))
                losses[name][str(strength)].append(loss)
                diagnostics.append(dict(inner=k, model=name, strength=strength, validation_log_loss=loss,
                                        iterations=fit["iterations"], max_abs_gradient=fit["max_abs_gradient"]))
        if progress:
            print(f"{progress}, inner {k + 1}/5 complete", flush=True)
    fitted = {}
    for name, (a, b, preprocessing) in prepared_pair(data, train, valid, models, noise, seed + 99).items():
        means = {s: float(np.mean(losses[name][str(s)])) for s in strengths}
        best = min(means.values())
        strength = max(s for s in strengths if means[s] <= best + 1e-8)
        fit = fit_logistic(a, y[train], strength, protocol["fit"]["optimizer_settings"])
        scores = predict(fit, b)
        fitted[name] = (fit, preprocessing, scores, dict(strength=strength, inner_mean_log_loss=means,
            validation_metrics=metrics(y[valid], scores), training_metrics=metrics(y[train], predict(fit, a)),
            max_abs_validation_feature=float(np.max(np.abs(b))), max_abs_validation_logit=float(np.max(np.abs(scores))),
            final_fit_iterations=fit["iterations"], max_abs_gradient=fit["max_abs_gradient"]))
    return fitted, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nested", "stability"], default="nested")
    args = ap.parse_args()
    directory = ROOT / f"audit/metacognition/{'development' if args.mode == 'nested' else 'stability'}-v0.3"
    directory.mkdir(exist_ok=False)
    amendment = json.loads((HERE / "amendment-v0.3.json").read_text())
    assert amendment["scale_floor"] == .1
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    report = dict(status="running", mode=args.mode, started_utc=datetime.now(timezone.utc).isoformat(),
        amendment_sha256=file_hash(HERE / "amendment-v0.3.json"),
        source_files={p.name: file_hash(p) for p in [Path(__file__), HERE / "probes.py", HERE / "features.py", HERE / "train_probes.py"]},
        calibration_partition_used=False, test_partition_used=False, fits=[], completed_outer_folds=[])
    write_json(directory / "report.json", report)
    start = time.perf_counter()
    try:
        protocol, split, data, cache, indices, y, fold, noise = load_training()
        models = amendment["development_models"] if args.mode == "nested" else amendment["stability"]["models"]
        report.update(models=models, split_sha256=file_hash(ROOT / "audit/metacognition/split-v0.2.json"),
                      feature_cache_manifest_sha256=file_hash(cache / "manifest.json"))
        scores = {m: np.full(len(y), np.nan) for m in models}
        repeats = 5 if args.mode == "nested" else amendment["stability"]["repeats"]
        rng = np.random.default_rng(amendment["stability"]["seed"])
        for k in range(repeats):
            if args.mode == "nested":
                train, valid = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                cv = stratified_folds(y[train], 20261027 + k)
            else:
                pool, valid = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
                train = np.concatenate([rng.choice(pool[y[pool] == c], len(pool[y[pool] == c]), replace=True) for c in (0, 1)])
                # Frozen assignment groups all copies of a sampled solution.
                cv = stratified_folds(y, 20261026)[train]
            fitted, diagnostics = select_and_fit(data, y, train, valid, cv, models, noise,
                20261025 + 100 * k, protocol, progress=f"{args.mode} {k + 1}/{repeats}")
            selection = {}
            for name, (fit, preprocessing, logits, diagnostic) in fitted.items():
                scores[name][valid] = logits
                np.savez_compressed(directory / f"outer-{k}-{name}.npz", weight=fit["weight"], intercept=np.array(fit["intercept"]),
                    validation_indices=indices[valid], validation_logits=logits, fitting_indices=indices[train], **preprocessing)
                selection[name] = diagnostic
            report["fits"].extend(dict(repeat=k, **d) for d in diagnostics)
            report["completed_outer_folds"].append(dict(fold=k, n_train=len(train), n_unique_train=len(set(train)),
                                                      n_validation=len(valid), selection=selection))
            write_json(directory / "report.json", report)
            if args.mode == "nested":
                np.savez_compressed(directory / "oof-progress.npz", indices=indices, failure=y, fold=fold, **scores)
        if args.mode == "nested":
            assert all(np.isfinite(s).all() for s in scores.values())
            report["results"] = {m: metrics(y, s) for m, s in scores.items()}
            report["pooled_oof_auc_difference"] = auc(y, scores["rich_hidden_primary"]) - auc(y, scores["rich"])
        else:
            differences = [r["selection"]["rich_hidden_primary"]["validation_metrics"]["auc"] - r["selection"]["rich"]["validation_metrics"]["auc"] for r in report["completed_outer_folds"]]
            report["delta_auc_sensitivity"] = dict(values=differences, mean=float(np.mean(differences)),
                median=float(np.median(differences)), range=[min(differences), max(differences)],
                descriptive_percentiles_2_5_97_5=np.quantile(differences, [.025, .975]).tolist(),
                caveat="Fixed development holdout fold 0; percentiles measure fitting-sample sensitivity, not inferential CI or statistical power")
        for name, digest in report["source_files"].items():
            assert file_hash(HERE / name) == digest
        assert file_hash(HERE / "amendment-v0.3.json") == report["amendment_sha256"]
        report.update(status="complete", wall_seconds=time.perf_counter() - start,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            artifacts={p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in directory.glob("*.npz")})
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        write_json(directory / "report.json", report)
        raise
    write_json(directory / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "wall_seconds", "results" if args.mode == "nested" else "delta_auc_sensitivity")}))


if __name__ == "__main__":
    main()
