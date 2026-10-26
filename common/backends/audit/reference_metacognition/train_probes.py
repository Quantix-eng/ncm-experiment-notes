"""Nested CV on the frozen training partition; never read held-out traces."""
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

from features import build_features
from probes import HiddenPCA, Standardizer, auc, fit_logistic, log_loss, metrics, predict, stratified_folds
from run_pilot import ROOT, file_hash, write_json

TRACE_KEYS = ("codes", "code_probs", "decoder_probs", "decoded_ids", "hidden_summary", "reconstruction_relative_change")


def feature_cache(directory, protocol, split):
    cache = ROOT / "audit/metacognition/train-features-v0.2"
    source_report = directory / "report.json"
    extraction = json.loads(source_report.read_text())
    if extraction["status"] != "complete":
        raise ValueError("Training extraction not complete")
    if cache.exists():
        metadata = json.loads((cache / "manifest.json").read_text())
        assert metadata["status"] == "complete"
        assert metadata["extraction_report_sha256"] == file_hash(source_report)
        assert metadata["features_code_sha256"] == file_hash(Path(__file__).with_name("features.py"))
        for name, fingerprint in metadata["files"].items():
            assert file_hash(cache / name) == fingerprint
    else:
        cache.mkdir()
        metadata = dict(status="building", extraction_report_sha256=file_hash(source_report),
                        features_code_sha256=file_hash(Path(__file__).with_name("features.py")),
                        protocol_sha256=split["protocol_sha256"], level=9)
        write_json(cache / "manifest.json", metadata)
        arrays = {}
        labels, indices, offset = [], [], 0
        for batch in extraction["batches"]:
            path = directory / batch["file"]
            assert file_hash(path) == batch["sha256"]
            with np.load(path, allow_pickle=False) as data:
                assert data["sample_indices"].tolist() == batch["indices"]
                trace = {key: data[key] for key in TRACE_KEYS}
                block = build_features(data["prefix"], trace, 9, protocol["level_lengths"])
                labels.extend(data["failure"].tolist())
                indices.extend(data["sample_indices"].tolist())
            if not arrays:
                for name, x in block.items():
                    arrays[name] = np.lib.format.open_memmap(cache / f"{name}.npy", mode="w+", dtype="float64", shape=(1000, x.shape[1]))
            size = len(block["rich"])
            for name, x in block.items():
                arrays[name][offset:offset + size] = x
            offset += size
        assert offset == 1000 and indices == split["partition_indices"]["train"] and sum(labels) == 137
        for x in arrays.values():
            x.flush()
        np.save(cache / "failure.npy", np.array(labels, dtype=np.int64))
        np.save(cache / "indices.npy", np.array(indices, dtype=np.int64))
        metadata.update(status="complete", shapes={name: list(x.shape) for name, x in arrays.items()},
                        files={p.name: file_hash(p) for p in cache.glob("*.npy")})
        write_json(cache / "manifest.json", metadata)
    return {name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("puzzle", "summary", "rich", "hidden_raw", "failure", "indices")}, cache


def prepared_pair(data, train, valid, models, noise, seed):
    output = {}
    for feature in ("rich", "puzzle", "summary"):
        relevant = [m for m in models if (feature == "rich" and m not in ("puzzle", "summary")) or m == feature]
        if not relevant:
            continue
        scaler = Standardizer().fit(np.asarray(data[feature][train]))
        a, b = scaler.transform(data[feature][train]), scaler.transform(data[feature][valid])
        if feature != "rich":
            output[feature] = (a, b, dict(base_mean=scaler.mean, base_keep=scaler.keep, base_scale=scaler.scale))
            continue
        for name in relevant:
            if name == "rich":
                output[name] = (a, b, dict(base_mean=scaler.mean, base_keep=scaler.keep, base_scale=scaler.scale))
            elif name == "rich_noise_control":
                ns = Standardizer().fit(noise[train])
                output[name] = (np.column_stack([a, ns.transform(noise[train])]),
                                np.column_stack([b, ns.transform(noise[valid])]),
                                dict(base_mean=scaler.mean, base_keep=scaler.keep, base_scale=scaler.scale,
                                     noise_mean=ns.mean, noise_keep=ns.keep, noise_scale=ns.scale))
            else:
                ht, hv = np.asarray(data["hidden_raw"][train]), np.asarray(data["hidden_raw"][valid])
                hidden_valid_rows = valid.copy()
                if name == "hidden_shuffle_control":
                    rng = np.random.default_rng(seed)
                    ht = ht[rng.permutation(len(ht))]
                    permutation = rng.permutation(len(hv))
                    hv, hidden_valid_rows = hv[permutation], valid[permutation]
                pca = HiddenPCA().fit(ht)
                output[name] = (np.column_stack([a, pca.transform(ht)]),
                                np.column_stack([b, pca.transform(hv)]),
                                dict(base_mean=scaler.mean, base_keep=scaler.keep, base_scale=scaler.scale,
                                     hidden_mean=pca.standardizer.mean, hidden_keep=pca.standardizer.keep,
                                     hidden_scale=pca.standardizer.scale, pca_axes=pca.axes,
                                     pca_score_mean=pca.score_scaler.mean, pca_score_scale=pca.score_scaler.scale,
                                     hidden_validation_source_rows=hidden_valid_rows))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["puzzle", "summary", "rich", "rich_hidden_primary", "rich_noise_control", "hidden_shuffle_control"],
                    default=["rich", "rich_hidden_primary", "rich_noise_control"])
    ap.add_argument("--output", type=Path, default=ROOT / "audit/metacognition/development-primary-v0.2")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol-v0.2.json")
    protocol = json.loads(protocol_path.read_text())
    split_path = ROOT / "audit/metacognition/split-v0.2.json"
    split = json.loads(split_path.read_text())
    assert file_hash(protocol_path) == split["protocol_sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        scope="nested_cross_validation_training_only_not_the_primary_held_out_test",
        protocol_sha256=file_hash(protocol_path), split_sha256=file_hash(split_path), models=args.models,
        source_files={p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("probes.py"), Path(__file__).with_name("features.py"))},
        procedure="outer frozen 5 folds; inner 5-fold stratified CV within each outer training set, all preprocessing refit; lambda selected by mean log loss",
        implementation_choices=dict(inner_seed="20261026+outer_fold+1", noise="seed 20261024 full array (2000,16), indexed by original sample index",
                                    hidden_shuffle="within inner/outer training and validation separately; never exchange across fold boundaries"),
        completed_outer_folds=[], fits=[], calibration_partition_used=False, test_partition_used=False,
        caveat="OOF scores come from five different fitted predictors; pooled AUC and descriptive resampling are exploratory; no final-test inference or confirmation power guarantee")
    write_json(args.output / "report.json", report)
    started = time.perf_counter()
    try:
        data, cache = feature_cache(ROOT / "audit/metacognition/train-v0.2", protocol, split)
        y, indices = np.asarray(data["failure"]), np.asarray(data["indices"])
        assert indices.tolist() == split["partition_indices"]["train"]
        fold = np.array([split["train_cv_fold"][str(i)] for i in indices])
        noise = np.random.default_rng(20261024).standard_normal((2000, 16))[indices]
        strengths = protocol["fit"]["lambda_grid"]
        scores = {model: np.full(len(y), np.nan) for model in args.models}
        baseline = np.full(len(y), np.nan)
        for outer in range(5):
            train, valid = np.flatnonzero(fold != outer), np.flatnonzero(fold == outer)
            inner = stratified_folds(y[train], 20261026 + outer + 1)
            losses = {name: {str(value): [] for value in strengths} for name in args.models}
            for inner_fold in range(5):
                it, iv = train[inner != inner_fold], train[inner == inner_fold]
                assert not (set(it) & set(valid)) and not (set(iv) & set(valid))
                prepared = prepared_pair(data, it, iv, args.models, noise, 20261025 + 100 * outer + inner_fold)
                for name, (a, b, _) in prepared.items():
                    for strength in strengths:
                        fitted = fit_logistic(a, y[it], strength, protocol["fit"]["optimizer_settings"])
                        loss = log_loss(y[iv], predict(fitted, b))
                        losses[name][str(strength)].append(loss)
                        report["fits"].append(dict(outer=outer, inner=inner_fold, model=name, strength=strength,
                            validation_log_loss=loss, iterations=fitted["iterations"], max_abs_gradient=fitted["max_abs_gradient"]))
                print(f"nested CV: outer {outer + 1}/5, inner {inner_fold + 1}/5 complete", flush=True)
            prepared = prepared_pair(data, train, valid, args.models, noise, 20261025 + 100 * outer + 99)
            selection = {}
            for name, (a, b, preprocessing) in prepared.items():
                means = {value: float(np.mean(losses[name][str(value)])) for value in strengths}
                best = min(means.values())
                strength = max(value for value in strengths if means[value] <= best + 1e-8)
                fitted = fit_logistic(a, y[train], strength, protocol["fit"]["optimizer_settings"])
                scores[name][valid] = predict(fitted, b)
                np.savez_compressed(args.output / f"outer-{outer}-{name}.npz", weight=fitted["weight"],
                                    intercept=np.array(fitted["intercept"]), validation_indices=indices[valid],
                                    validation_logits=scores[name][valid], **preprocessing)
                selection[name] = dict(strength=strength, inner_mean_log_loss={str(k): v for k, v in means.items()},
                                       validation_metrics=metrics(y[valid], scores[name][valid]),
                                       training_metrics=metrics(y[train], predict(fitted, a)),
                                       final_fit_iterations=fitted["iterations"], max_abs_gradient=fitted["max_abs_gradient"])
            base_p = y[train].mean()
            baseline[valid] = np.log(base_p / (1 - base_p))
            report["completed_outer_folds"].append(dict(fold=outer, n_train=len(train), n_validation=len(valid), selection=selection))
            np.savez_compressed(args.output / "oof-progress.npz", indices=indices, failure=y, fold=fold, constant=baseline, **scores)
            write_json(args.output / "report.json", report)
        assert all(np.isfinite(score).all() for score in scores.values())
        report.update(status="complete", n=1000, failures=int(y.sum()),
                      results={name: metrics(y, score) for name, score in scores.items()},
                      constant_oof=metrics(y, baseline), feature_cache_manifest_sha256=file_hash(cache / "manifest.json"),
                      wall_seconds=time.perf_counter() - started,
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        if "rich" in scores and "rich_hidden_primary" in scores:
            report["pooled_oof_auc_difference"] = auc(y, scores["rich_hidden_primary"]) - auc(y, scores["rich"])
            report["within_class_score_correlations"] = {str(label): float(np.corrcoef(scores["rich"][y == label], scores["rich_hidden_primary"][y == label])[0, 1]) for label in (0, 1)}
        for name, fingerprint in report["source_files"].items():
            assert file_hash(Path(__file__).with_name(name)) == fingerprint
        report["artifacts"] = {p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in args.output.glob("*.npz")}
    except Exception:
        report.update(status="failed", error=traceback.format_exc(), wall_seconds=time.perf_counter() - started)
        write_json(args.output / "report.json", report)
        raise
    write_json(args.output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "results", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
