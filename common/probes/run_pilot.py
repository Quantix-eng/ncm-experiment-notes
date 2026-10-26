"""Freeze the split, then extract a training-only pilot with exact parity gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from ncm_mac.sudoku import SudokuModel, load_boards, verify_assets
from design import canonical_hash, cv_folds, partitions, pilot_indices
from features import CELL_POS, build_features, observed_counts, prompt_cells
from trace import instrumented_prediction


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for owner, module in (("generator", model.net), ("vqvae", model.vqvae)):
        for name, tensor in sorted(module.state_dict().items()):
            h.update(f"{owner}/{name}/{tensor.dtype}/{list(tensor.shape)}".encode())
            h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def assert_prediction_equal(a, b):
    for field in ("codes", "reconstruction", "logits", "ids"):
        if not torch.equal(getattr(a, field), getattr(b, field)):
            raise AssertionError(f"Bitwise mismatch in {field}")


def source_fingerprints():
    paths = list((ROOT / "src/ncm_mac").rglob("*.py"))
    paths += list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).with_name("protocol-v0.2.json")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def prepare(protocol):
    verify_assets(ROOT, "hard", include_data=True)
    directory = ROOT / "data/sudoku/hard"
    puzzles = load_boards(directory / "puzzles.pickle")
    solutions = load_boards(directory / "solutions.pickle")
    pred_path = ROOT / "audit/sudoku-parity-full-hard-predictions.json"
    predictions = json.loads(pred_path.read_text())
    archived = json.loads((ROOT / "audit/sudoku-parity-full.json").read_text())["tasks"]["hard"]
    actual_prediction_hash = hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest()
    if actual_prediction_hash != archived["predictions_sha256"]:
        raise AssertionError("Historical predictions integrity failure")
    if archived["official_sample_indices"] != list(range(2000)) or len(puzzles) != 2000:
        raise AssertionError("Expected exhaustive ordered hard archive")
    failure = [int(pred != ref) for pred, ref in zip(predictions, solutions)]
    if len(failure) != 2000 or sum(failure) != 274:
        raise AssertionError("Unexpected historical outcomes")
    groups = [canonical_hash(s) for s in solutions]
    split = partitions(failure, groups, seed=protocol["split"]["seed"])
    pilot = pilot_indices(split["train"], failure)
    if not set(pilot).issubset(split["train"]):
        raise AssertionError("Pilot leakage")
    # Easy/medium cannot be treated as independently held-out solutions later.
    for difficulty in ("easy", "medium"):
        other = load_boards(ROOT / f"data/sudoku/{difficulty}/solutions.pickle")
        if other != solutions:
            raise AssertionError("Cross-difficulty grouping assumption changed")
    manifest = dict(protocol_sha256=file_hash(Path(__file__).with_name("protocol-v0.2.json")),
        grouping="exact hard solution hash; easy/medium same index inherits partition; no exhaustive isomorphism audit",
        design_code_sha256=file_hash(Path(__file__).with_name("design.py")),
        source_archived_report_sha256=file_hash(ROOT / "audit/sudoku-parity-full.json"),
        source_predictions_sha256=file_hash(pred_path),
        assets_lock_sha256=file_hash(ROOT / "assets.lock.json"),
        partition_indices=split, solution_group_sha256=groups,
        class_counts={name: dict(n=len(idx), failures=sum(failure[i] for i in idx),
                                 successes=sum(not failure[i] for i in idx)) for name, idx in split.items()},
        train_cv_fold=cv_folds(split["train"], failure),
        pilot_indices=pilot, pilot_seed=20261028, pilot_class_counts=dict(n=32, failures=8, successes=24),
        full_trace_sample_indices=pilot[:2],
        feature_evaluation_boundary="gold solutions only in split stratification and separate evaluation, never passed to trace/features",
        confirmation_status="designed in protocol; no independent corpus generated")
    dest = ROOT / "audit/metacognition/split-v0.2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if json.loads(dest.read_text()) != manifest:
            raise AssertionError("Existing frozen split differs; refusing overwrite")
    else:
        write_json(dest, manifest)
    return manifest, puzzles, solutions, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--output", type=Path, default=ROOT / "audit/metacognition/pilot-v0.2")
    args = ap.parse_args()
    protocol = json.loads(Path(__file__).with_name("protocol-v0.2.json").read_text())
    torch.set_num_threads(protocol["reference"]["threads"])
    torch.manual_seed(20261023)
    manifest, puzzles, solutions, historical = prepare(protocol)
    print(json.dumps({"split_counts": manifest["class_counts"], "pilot": manifest["pilot_class_counts"]}), flush=True)
    if args.prepare_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        scope="training-only intentionally enriched technical pilot; not representative performance or probe evaluation",
        protocol_sha256=manifest["protocol_sha256"],
        split_sha256=file_hash(ROOT / "audit/metacognition/split-v0.2.json"),
        source_files=source_fingerprints(),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         platform=platform.platform(), threads=torch.get_num_threads(), device="cpu", dtype="float32"),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        disk_free_before=shutil.disk_usage(ROOT).free, batch_size=8,
        batches=[], excluded_partitions=["calibration", "test"], selected_indices=manifest["pilot_indices"])
    write_json(args.output / "report.json", report)
    try:
        model = SudokuModel(ROOT / "artifacts/sudoku/hard")
        if list(model.net.levels) != protocol["level_lengths"] or len(model.net.transformer.blocks) != 8:
            raise AssertionError("Checkpoint architecture not as specified")
        original_state = state_hash(model)
        hook_count_before = sum(len(m._forward_hooks) for m in model.net.modules())
        selected = manifest["pilot_indices"]
        prefix_all = model.prefix([puzzles[i] for i in selected])
        prompt_cells(prefix_all.numpy())
        model.predict(prefix_all[:8])  # warmup, explicitly not part of timings
        pieces, raw_path = [], args.output / "full-traces-2.npz"
        for start in range(0, 32, 8):
            prefix = prefix_all[start:start + 8]
            t0 = time.perf_counter()
            reference = model.predict(prefix)
            baseline_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            observed, trace, raw, timing = instrumented_prediction(model, prefix)
            extraction_wall = time.perf_counter() - t0
            assert_prediction_equal(reference, observed)
            # A separate unhooked, uncached eager path at the identical batch.
            eager = model.predict(prefix, optimized=False)
            assert_prediction_equal(eager, observed)
            if observed.ids.tolist() != [historical[i] for i in selected[start:start + 8]]:
                raise AssertionError("Historical answer differs at this batching")
            if sum(len(m._forward_hooks) for m in model.net.modules()) != hook_count_before:
                raise AssertionError("Leaked observation hooks")
            if start == 0:
                np.savez_compressed(raw_path,
                    sample_indices=np.array(selected[:2]),
                    hidden=np.concatenate([h[:2].numpy() for h in raw["hidden"]], axis=1),
                    code_logits=np.concatenate([x[:2].numpy() for x in raw["code_logits"]], axis=1),
                    reconstruction=np.stack([x[:2].numpy() for x in raw["reconstruction"]], axis=1),
                    decoder_logits=np.stack([x[:2].numpy() for x in raw["decoder_logits"]], axis=1))
            pieces.append(trace)
            report["batches"].append(dict(indices=selected[start:start + 8],
                reference_cached_seconds=baseline_time, full_extraction_seconds=extraction_wall,
                bitwise_cached_and_eager_equal=True, historical_answers_equal=True,
                snapshots_independent_and_replay_equal=True, hooks_and_cache_cleared=True, **timing))
            print(f"pilot: {start + 8}/32; cached/eager/captured bitwise equal", flush=True)
        trace = {name: np.concatenate([part[name] for part in pieces]) for name in pieces[0]}
        features = build_features(prefix_all.numpy(), trace, 9, protocol["level_lengths"])
        # Runtime check of the temporal boundary, independently of the unit test.
        poisoned = {name: value.copy() for name, value in trace.items()}
        boundary = sum(protocol["level_lengths"][:9])
        poisoned["code_probs"][:, boundary:] = np.nan
        poisoned["codes"][:, boundary:] = 99
        for name in ("decoder_probs", "hidden_summary", "reconstruction_relative_change"):
            poisoned[name][:, 9:] = np.nan
        poisoned["decoded_ids"][:, 9:] = 99
        protected = build_features(prefix_all.numpy(), poisoned, 9, protocol["level_lengths"])
        if not all(np.array_equal(features[k], protected[k]) for k in features):
            raise AssertionError("Future-state leakage")
        if state_hash(model) != original_state:
            raise AssertionError("Model parameters or persistent buffers mutated")
        if source_fingerprints() != report["source_files"]:
            raise AssertionError("Source changed during pilot")
        trace_path = args.output / "traces-32.npz"
        np.savez_compressed(trace_path, sample_indices=np.array(selected), prefix=prefix_all.numpy(), **trace)
        refs = np.array([solutions[i] for i in selected])
        given = prompt_cells(prefix_all.numpy())
        level_diagnostics = []
        for k in range(12):
            ids = trace["decoded_ids"][:, k]
            invalid, bad_sep, clue_bad, pairs, valid = observed_counts(ids, given)
            correct = ids[:, CELL_POS] == refs[:, CELL_POS]
            level_diagnostics.append(dict(level=k + 1, exact_successes=int(np.all(ids == refs, axis=1).sum()),
                valid_grids_with_clues=int(valid.sum()), malformed_outputs=int(((invalid + bad_sep) != 0).sum()),
                correct_cell_count=int(correct.sum()), total_cells=32 * 81,
                clue_violations=int(clue_bad.sum()), duplicate_pairs=int(pairs.sum())))
        write_json(args.output / "evaluation-training-only.json", dict(
            warning="Technical sample selected using historical outcome: 24 successes/8 failures by design; do not estimate corpus rates",
            sample_indices=selected, levels=level_diagnostics,
            gold_cell_correctness=(trace["decoded_ids"][:, :, CELL_POS] == refs[:, None, CELL_POS]).astype(int).tolist()))
        uncompressed = sum(x.nbytes for x in trace.values()) + prefix_all.numpy().nbytes
        report.update(status="complete", model_state_sha256_before_and_after=original_state,
            final_successes=level_diagnostics[-1]["exact_successes"], final_failures=32 - level_diagnostics[-1]["exact_successes"],
            predictor_shapes={name: list(value.shape) for name, value in features.items()},
            finite_predictors=True, future_poison_check_passed=True,
            trace_arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in trace.items()},
            uncompressed_trace_bytes=uncompressed, projected_uncompressed_trace_bytes_2000=uncompressed / 32 * 2000,
            full_traces_selected_before_inference=selected[:2],
            relative_reconstruction_change_level1="zero sentinel: no previous state, not a meaningful relative difference",
            total_wall_seconds=time.perf_counter() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            memory_scope="whole process including model, captures, references and verification buffers; not single-model standalone memory",
            run_artifacts={p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in args.output.iterdir() if p.name != "report.json"},
            confirmation_status="not generated; no sample-size guarantee", probe_training="not started")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        write_json(args.output / "report.json", report)
        raise
    write_json(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "final_successes", "final_failures", "predictor_shapes", "total_wall_seconds", "process_peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
