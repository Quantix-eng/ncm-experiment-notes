"""Prespecified levels 3/6/11: secondary analyses, never replace level9.

Primary outcomes have already been seen. The fixed amended method is applied
unchanged; each secondary model is locked before new held-out extraction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from features import build_features
from final_v03 import transformed
from probes import auc, fit_logistic, log_loss, metrics, predict
from run_pilot import ROOT, SudokuModel, assert_prediction_equal, file_hash, load_boards, state_hash, write_json
from study_v03 import HERE, load_training, prepared_pair
from trace import instrumented_prediction
from train_probes import TRACE_KEYS

DIRECTORY = ROOT / "audit/metacognition/sensitivity-v0.3"
LEVELS, MODELS = [3, 6, 11], ["rich", "rich_hidden_primary", "summary"]


def sources():
    return {name: file_hash(HERE / name) for name in ("sensitivity_v03.py", "study_v03.py", "probes.py", "features.py", "final_v03.py", "trace.py")}


def fit():
    DIRECTORY.mkdir(exist_ok=False)
    protocol, split, _, _, indices, y, fold, noise = load_training()
    choices = {}
    extraction = ROOT / "audit/metacognition/train-v0.2"
    report = json.loads((extraction / "report.json").read_text())
    for level in LEVELS:
        parts = {}
        for batch in report["batches"]:
            path = extraction / batch["file"]
            assert file_hash(path) == batch["sha256"]
            with np.load(path, allow_pickle=False) as trace:
                block = build_features(trace["prefix"], {k: trace[k] for k in TRACE_KEYS}, level, protocol["level_lengths"])
                for name, value in block.items():
                    parts.setdefault(name, []).append(value)
        data = {name: np.concatenate(values) for name, values in parts.items()}
        del parts
        losses = {m: {s: [] for s in protocol["fit"]["lambda_grid"]} for m in MODELS}
        for k in range(5):
            train, valid = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            for name, (a, b, _) in prepared_pair(data, train, valid, MODELS, noise, 20261101 + k).items():
                for strength in losses[name]:
                    fitted = fit_logistic(a, y[train], strength, protocol["fit"]["optimizer_settings"])
                    losses[name][strength].append(log_loss(y[valid], predict(fitted, b)))
            print(f"secondary level{level}, CV {k + 1}/5", flush=True)
        selected = {}
        for name, (a, _, preprocessing) in prepared_pair(data, np.arange(len(y)), np.array([0]), MODELS, noise, 20261109).items():
            means = {s: float(np.mean(values)) for s, values in losses[name].items()}
            strength = max(s for s in means if means[s] <= min(means.values()) + 1e-8)
            fitted = fit_logistic(a, y, strength, protocol["fit"]["optimizer_settings"])
            preprocessing.pop("hidden_validation_source_rows", None)
            np.savez_compressed(DIRECTORY / f"model-{level}-{name}.npz", weight=fitted["weight"],
                                intercept=np.array(fitted["intercept"]), **preprocessing)
            selected[name] = dict(strength=strength, cv_log_loss=means)
        choices[str(level)] = selected
    write_json(DIRECTORY / "lock.json", dict(status="locked", levels=LEVELS, models=MODELS, choices=choices,
        source_files=sources(), parent_primary_lock_sha256=file_hash(ROOT / "audit/metacognition/final-v0.3/lock.json"),
        amendment_sha256=file_hash(HERE / "amendment-v0.3.json"),
        models_sha256={p.name: file_hash(p) for p in DIRECTORY.glob("model-*.npz")},
        disclosure="Prespecified secondary levels executed after seeing level9 results; same method, no level selection, no replacement of primary; no multiplicity-adjusted confirmatory claims"))


def locked():
    lock = json.loads((DIRECTORY / "lock.json").read_text())
    assert lock["source_files"] == sources()
    for name, digest in lock["models_sha256"].items():
        assert file_hash(DIRECTORY / name) == digest
    return lock


def extract():
    locked()
    destination = DIRECTORY / "held-out"
    destination.mkdir(exist_ok=False)
    protocol = json.loads((HERE / "protocol-v0.2.json").read_text())
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    puzzles = load_boards(ROOT / "data/sudoku/hard/puzzles.pickle")
    labels = load_boards(ROOT / "data/sudoku/hard/solutions.pickle")
    historical = json.loads((ROOT / "audit/sudoku-parity-full-hard-predictions.json").read_text())
    model = SudokuModel(ROOT / "artifacts/sudoku/hard")
    before = state_hash(model)
    report = dict(status="running", lock_sha256=file_hash(DIRECTORY / "lock.json"), batches=[])
    for partition in ("calibration", "test"):
        selected = split["partition_indices"][partition]
        parts = {level: {} for level in LEVELS}
        for offset in range(0, len(selected), 8):
            indices = selected[offset:offset + 8]
            prefix = model.prefix([puzzles[i] for i in indices])
            reference = model.predict(prefix)
            prediction, trace, raw, _ = instrumented_prediction(model, prefix)
            assert_prediction_equal(reference, prediction)
            assert prediction.ids.tolist() == [historical[i] for i in indices]
            for level in LEVELS:
                block = build_features(prefix.numpy(), {k: trace[k] for k in TRACE_KEYS}, level, protocol["level_lengths"])
                for name, values in block.items():
                    parts[level].setdefault(name, []).append(values)
            del reference, prediction, raw, trace
            report["batches"].append(dict(partition=partition, indices=indices, cached_bitwise_equal=True, historical_answers_equal=True))
            if offset % 80 == 0:
                print(f"secondary extraction {partition} {offset + len(indices)}/500", flush=True)
        for level in LEVELS:
            np.savez_compressed(destination / f"{partition}-{level}.npz", indices=np.array(selected),
                failure=np.array([int(historical[i] != labels[i]) for i in selected]),
                **{name: np.concatenate(value) for name, value in parts[level].items()})
    assert state_hash(model) == before
    locked()
    report.update(status="complete", model_state_sha256_before_and_after=before,
                  artifacts={p.name: file_hash(p) for p in destination.glob("*.npz")})
    write_json(destination / "report.json", report)


def evaluate():
    lock = locked()
    destination = DIRECTORY / "analysis"
    destination.mkdir(exist_ok=False)
    extraction = json.loads((DIRECTORY / "held-out/report.json").read_text())
    assert extraction["status"] == "complete"
    for name, digest in extraction["artifacts"].items():
        assert file_hash(DIRECTORY / "held-out" / name) == digest
    settings = json.loads((HERE / "protocol-v0.2.json").read_text())["fit"]["optimizer_settings"]
    outputs = {}
    for level in LEVELS:
        platt, scores, results = {}, {}, {}
        with np.load(DIRECTORY / f"held-out/calibration-{level}.npz", allow_pickle=False) as data:
            for name in MODELS:
                with np.load(DIRECTORY / f"model-{level}-{name}.npz", allow_pickle=False) as model:
                    score = transformed(model, data, name) @ model["weight"] + model["intercept"]
                fitted = fit_logistic(score[:, None], data["failure"], .001, settings)
                platt[name] = dict(slope=float(fitted["weight"][0]), intercept=fitted["intercept"])
        write_json(destination / f"calibration-{level}.json", platt)
        with np.load(DIRECTORY / f"held-out/test-{level}.npz", allow_pickle=False) as data:
            y = data["failure"]
            for name in MODELS:
                with np.load(DIRECTORY / f"model-{level}-{name}.npz", allow_pickle=False) as model:
                    x = transformed(model, data, name)
                    scores[name] = x @ model["weight"] + model["intercept"]
                adjusted = scores[name] * platt[name]["slope"] + platt[name]["intercept"]
                results[name] = dict(raw=metrics(y, scores[name]), calibrated=metrics(y, adjusted),
                                     max_abs_feature=float(np.max(np.abs(x))), max_abs_logit=float(np.max(np.abs(scores[name]))))
            np.savez_compressed(destination / f"test-predictions-{level}.npz", indices=data["indices"], failure=y, **scores)
        rng = np.random.default_rng(20261027)
        p, n = np.flatnonzero(y), np.flatnonzero(y == 0)
        deltas = []
        for _ in range(5000):
            rows = np.r_[rng.choice(p, len(p), replace=True), rng.choice(n, len(n), replace=True)]
            deltas.append(auc(y[rows], scores["rich_hidden_primary"][rows]) - auc(y[rows], scores["rich"][rows]))
        outputs[str(level)] = dict(results=results, delta_auc=auc(y, scores["rich_hidden_primary"]) - auc(y, scores["rich"]),
                                   descriptive_bootstrap_95=np.quantile(deltas, [.025, .975]).tolist())
    write_json(destination / "report.json", dict(status="complete", lock_sha256=file_hash(DIRECTORY / "lock.json"),
        disclosure=lock["disclosure"], results=outputs, primary_level9_unchanged=True,
        caveat="Secondary sensitivity analyses, no familywise multiplicity adjustment; raw logits for AUROC; conditional fixed-predictor bootstrap",
        artifacts={p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps({level: {k:v for k,v in result.items() if k != "results"} for level,result in outputs.items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "extract", "evaluate"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    {"fit": fit, "extract": extract, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
