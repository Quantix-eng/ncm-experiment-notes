"""Fit/lock, extract held-out traces, then evaluate exactly once under v0.3.

Stages are deliberately separate: a hash lock must exist before new held-out
inference; Platt calibration precedes test reporting. No auto-retuning.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from features import build_features
from probes import auc, fit_logistic, log_loss, metrics, predict
from run_pilot import (ROOT, SudokuModel, assert_prediction_equal, file_hash,
                       load_boards, state_hash, verify_assets, write_json)
from study_v03 import HERE, load_training, prepared_pair
from trace import instrumented_prediction
from train_probes import TRACE_KEYS

DIRECTORY = ROOT / "audit/metacognition/final-v0.3"
MODELS = ["rich", "rich_hidden_primary", "summary", "rich_noise_control"]


def fingerprints():
    files = [HERE / name for name in ("final_v03.py", "study_v03.py", "probes.py", "features.py", "trace.py",
              "run_pilot.py", "train_probes.py", "amendment-v0.3.json", "protocol-v0.2.json",
              "stability_v03.py", "stability-correction-v0.3.json")]
    files += list((ROOT / "src/ncm_mac").rglob("*.py"))
    files += [ROOT / "assets.lock.json", ROOT / "audit/metacognition/split-v0.2.json"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def locked():
    lock = json.loads((DIRECTORY / "lock.json").read_text())
    assert lock["status"] == "locked_before_new_held_out_inference"
    for name, digest in lock["source_files"].items():
        assert file_hash(ROOT / name) == digest, name
    for name, digest in lock["model_files"].items():
        assert file_hash(DIRECTORY / name) == digest
    return lock


def fit_and_lock():
    for name in ("development", "stability"):
        source = ROOT / f"audit/metacognition/{name}-v0.3/report.json"
        r = json.loads(source.read_text())
        assert r["status"] == "complete"
        assert r["amendment_sha256"] == file_hash(HERE / "amendment-v0.3.json")
        verification = json.loads(source.with_name("verification.json").read_text())
        assert verification["status"] == "passed" and verification["report_sha256"] == file_hash(source)
        for filename, digest in r["source_files"].items():
            assert file_hash(HERE / filename) == digest
        # Safety gate is numerical, not an outcome-based feature-selection rule.
        for repeat in r["completed_outer_folds"]:
            for model in ("rich", "rich_hidden_primary"):
                assert repeat["selection"][model]["max_abs_validation_logit"] < 1000
    DIRECTORY.mkdir(exist_ok=False)
    protocol, split, data, cache, indices, y, fold, noise = load_training()
    losses = {m: {str(s): [] for s in protocol["fit"]["lambda_grid"]} for m in MODELS}
    for k in range(5):
        train, valid = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        for name, (a, b, _) in prepared_pair(data, train, valid, MODELS, noise, 20261101 + k).items():
            for strength in protocol["fit"]["lambda_grid"]:
                fit = fit_logistic(a, y[train], strength, protocol["fit"]["optimizer_settings"])
                losses[name][str(strength)].append(log_loss(y[valid], predict(fit, b)))
        print(f"final fitting CV {k + 1}/5", flush=True)
    choices = {}
    all_rows = np.arange(len(y))
    # One dummy TRAINING row supplies a transform output, which is discarded;
    # the final fit uses all training rows, no held-out feature or label.
    for name, (a, _, preprocessing) in prepared_pair(data, all_rows, all_rows[:1], MODELS, noise, 20261109).items():
        means = {float(s): float(np.mean(v)) for s, v in losses[name].items()}
        best = min(means.values())
        strength = max(s for s in means if means[s] <= best + 1e-8)
        fit = fit_logistic(a, y, strength, protocol["fit"]["optimizer_settings"])
        preprocessing.pop("hidden_validation_source_rows", None)
        np.savez_compressed(DIRECTORY / f"model-{name}.npz", weight=fit["weight"], intercept=np.array(fit["intercept"]), **preprocessing)
        choices[name] = dict(strength=strength, cv_log_loss=means, iterations=fit["iterations"], max_abs_gradient=fit["max_abs_gradient"])
    lock = dict(status="locked_before_new_held_out_inference", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        primary="AUROC(raw failure logit rich_hidden_primary) - AUROC(raw failure logit rich), level9, test500",
        interpretation="exploratory held-out evaluation after development amendment; not prospective confirmation; historical labels previously seen for stratification",
        n_test=500, expected_test_failures=68, models=MODELS, choices=choices,
        bootstrap=dict(seed=20261027, repetitions=5000, method="paired stratified percentile by unique solution group", interval=.95),
        practical_delta=.03, sample_size_rule="fixed original test500; no optional extension or 80% power claim",
        calibration="Platt, lambda .001, calibration500 only; raw logits for AUROC",
        source_files=fingerprints(), model_files={p.name: file_hash(p) for p in DIRECTORY.glob("model-*.npz")},
        development_report_sha256=file_hash(ROOT / "audit/metacognition/development-v0.3/report.json"),
        stability_report_sha256=file_hash(ROOT / "audit/metacognition/stability-v0.3/report.json"))
    write_json(DIRECTORY / "lock.json", lock)
    print(json.dumps(dict(status=lock["status"], choices=choices)))


def extract():
    lock = locked()
    destination = DIRECTORY / "held-out"
    destination.mkdir(exist_ok=False)
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    protocol = json.loads((HERE / "protocol-v0.2.json").read_text())
    verify_assets(ROOT, "hard", include_data=True)
    puzzles = load_boards(ROOT / "data/sudoku/hard/puzzles.pickle")
    solutions = load_boards(ROOT / "data/sudoku/hard/solutions.pickle")
    historical = json.loads((ROOT / "audit/sudoku-parity-full-hard-predictions.json").read_text())
    model = SudokuModel(ROOT / "artifacts/sudoku/hard")
    before = state_hash(model)
    report = dict(status="running", lock_sha256=file_hash(DIRECTORY / "lock.json"), batches=[], partition_indices={}, level=9)
    start = time.perf_counter()
    for partition in ("calibration", "test"):
        selected = split["partition_indices"][partition]
        assert not set(selected) & set(split["partition_indices"]["train"])
        parts, labels = {}, []
        for number, offset in enumerate(range(0, len(selected), 8)):
            indices = selected[offset:offset + 8]
            prefix = model.prefix([puzzles[i] for i in indices])
            reference = model.predict(prefix)
            prediction, trace, raw, timing = instrumented_prediction(model, prefix)
            assert_prediction_equal(reference, prediction)
            eager_checked = number % 10 == 0 or offset + 8 >= len(selected)
            if eager_checked:
                assert_prediction_equal(model.predict(prefix, optimized=False), prediction)
            assert prediction.ids.tolist() == [historical[i] for i in indices]
            # Keep labels outside feature construction; no labels in trace keys.
            features = build_features(prefix.numpy(), {k: trace[k] for k in TRACE_KEYS}, 9, protocol["level_lengths"])
            for name, values in features.items():
                parts.setdefault(name, []).append(values)
            labels.extend(int(historical[i] != solutions[i]) for i in indices)
            report["batches"].append(dict(partition=partition, indices=indices, cached_bitwise_equal=True,
                eager_bitwise_checked=eager_checked, historical_answers_equal=True))
            del reference, prediction, trace, raw
            if number % 10 == 0:
                print(f"locked extraction {partition} {offset + len(indices)}/500", flush=True)
        np.savez_compressed(destination / f"{partition}-features.npz", indices=np.array(selected), failure=np.array(labels),
                            **{name: np.concatenate(values) for name, values in parts.items()})
        report["partition_indices"][partition] = selected
        write_json(destination / "report.json", report)
    assert state_hash(model) == before
    locked()
    report.update(status="complete", model_state_sha256_before_and_after=before, wall_seconds=time.perf_counter() - start,
                  artifacts={p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in destination.glob("*.npz")})
    write_json(destination / "report.json", report)


def transformed(saved, data, name):
    family = "summary" if name == "summary" else "rich"
    base = data[family]
    x = (base[:, saved["base_keep"]] - saved["base_mean"][saved["base_keep"]]) / saved["base_scale"]
    if name == "rich_hidden_primary":
        h = data["hidden_raw"]
        h = (h[:, saved["hidden_keep"]] - saved["hidden_mean"][saved["hidden_keep"]]) / saved["hidden_scale"]
        h = (h @ saved["pca_axes"].T - saved["pca_score_mean"]) / saved["pca_score_scale"]
        x = np.column_stack([x, h])
    elif name == "rich_noise_control":
        noise = np.random.default_rng(20261024).standard_normal((2000, 16))[data["indices"]]
        noise = (noise[:, saved["noise_keep"]] - saved["noise_mean"][saved["noise_keep"]]) / saved["noise_scale"]
        x = np.column_stack([x, noise])
    return x


def probability_diagnostics(y, logit):
    probability = 1 / (1 + np.exp(-np.clip(logit, -700, 700)))
    bins = []
    for k in range(10):
        selected = (probability >= k / 10) & ((probability < (k + 1) / 10) if k < 9 else (probability <= 1))
        bins.append(dict(lower=k / 10, upper=(k + 1) / 10, n=int(selected.sum()),
                         mean_probability=float(probability[selected].mean()) if selected.any() else None,
                         failure_rate=float(y[selected].mean()) if selected.any() else None))
    order = np.argsort(logit, kind="stable")
    curve = [dict(coverage=k / 10, retained=k * len(y) // 10,
                  failure_risk=float(y[order[:k * len(y) // 10]].mean())) for k in range(1, 11)]
    return dict(reliability_bins=bins, risk_coverage=curve)


def evaluate():
    lock = locked()
    destination = DIRECTORY / "analysis"
    destination.mkdir(exist_ok=False)
    extraction = json.loads((DIRECTORY / "held-out/report.json").read_text())
    assert extraction["status"] == "complete" and extraction["lock_sha256"] == file_hash(DIRECTORY / "lock.json")
    for name, entry in extraction["artifacts"].items():
        assert file_hash(DIRECTORY / "held-out" / name) == entry["sha256"]
    settings = json.loads((HERE / "protocol-v0.2.json").read_text())["fit"]["optimizer_settings"]
    platt, calibration_metrics = {}, {}
    with np.load(DIRECTORY / "held-out/calibration-features.npz", allow_pickle=False) as data:
        y = data["failure"]
        for name in MODELS:
            with np.load(DIRECTORY / f"model-{name}.npz", allow_pickle=False) as model:
                logit = transformed(model, data, name) @ model["weight"] + model["intercept"]
            fit = fit_logistic(logit[:, None], y, .001, settings)
            platt[name] = dict(slope=float(fit["weight"][0]), intercept=fit["intercept"])
            calibration_metrics[name] = dict(raw=metrics(y, logit), calibrated=metrics(y, predict(fit, logit[:, None])))
    # Calibration parameters are materialized before reading test features/labels.
    write_json(destination / "calibration.json", dict(lock_sha256=file_hash(DIRECTORY / "lock.json"), platt=platt, metrics=calibration_metrics))
    raw, results = {}, {}
    with np.load(DIRECTORY / "held-out/test-features.npz", allow_pickle=False) as data:
        y, indices = data["failure"], data["indices"]
        assert len(y) == lock["n_test"] and int(y.sum()) == lock["expected_test_failures"]
        for name in MODELS:
            with np.load(DIRECTORY / f"model-{name}.npz", allow_pickle=False) as model:
                x = transformed(model, data, name)
                raw[name] = x @ model["weight"] + model["intercept"]
            calibrated = raw[name] * platt[name]["slope"] + platt[name]["intercept"]
            results[name] = dict(raw=metrics(y, raw[name]), calibrated=metrics(y, calibrated),
                                 max_abs_feature=float(np.max(np.abs(x))), max_abs_raw_logit=float(np.max(np.abs(raw[name]))),
                                 diagnostics=probability_diagnostics(y, calibrated))
        np.savez_compressed(destination / "test-predictions.npz", indices=indices, failure=y, **raw)
    rng = np.random.default_rng(lock["bootstrap"]["seed"])
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    differences = []
    for _ in range(lock["bootstrap"]["repetitions"]):
        rows = np.r_[rng.choice(p, len(p), replace=True), rng.choice(n, len(n), replace=True)]
        differences.append(auc(y[rows], raw["rich_hidden_primary"][rows]) - auc(y[rows], raw["rich"][rows]))
    interval = np.quantile(differences, [.025, .975]).tolist()
    difference = auc(y, raw["rich_hidden_primary"]) - auc(y, raw["rich"])
    report = dict(status="complete", lock_sha256=file_hash(DIRECTORY / "lock.json"), n=len(y), failures=int(y.sum()),
                  results=results, primary_delta_auc=difference, primary_paired_bootstrap_95=interval,
                  evidence_above_zero=interval[0] > 0, upper_bound_below_practical_0_03=interval[1] < .03,
                  caveat="Exploratory held-out result under amendment v0.3; fixed-predictor conditional bootstrap, not independent confirmation or native metacognition proof",
                  artifacts={p.name: file_hash(p) for p in destination.iterdir() if p.is_file()})
    locked()
    write_json(destination / "report.json", report)
    print(json.dumps(dict(status=report["status"], delta=difference, interval=interval, results={m:r["raw"] for m,r in results.items()})))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "extract", "evaluate"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    {"fit": fit_and_lock, "extract": extract, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
