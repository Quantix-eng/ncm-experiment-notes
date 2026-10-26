"""Extract only the frozen training partition into restartable numeric shards."""
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

from run_pilot import (ROOT, SudokuModel, assert_prediction_equal, file_hash,
                       load_boards, source_fingerprints, state_hash, verify_assets, write_json)
from trace import instrumented_prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "audit/metacognition/train-v0.2")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    split_path = ROOT / "audit/metacognition/split-v0.2.json"
    split = json.loads(split_path.read_text())
    protocol_path = Path(__file__).with_name("protocol-v0.2.json")
    assert file_hash(protocol_path) == split["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    selected = split["partition_indices"]["train"]
    assert len(selected) == 1000 and not (set(selected) & set(split["partition_indices"]["test"]))
    assert not (set(selected) & set(split["partition_indices"]["calibration"]))
    verify_assets(ROOT, "hard", include_data=True)
    puzzles = load_boards(ROOT / "data/sudoku/hard/puzzles.pickle")
    # Restrict labels/references to training indices immediately after loading.
    references = load_boards(ROOT / "data/sudoku/hard/solutions.pickle")
    references = {i: references[i] for i in selected}
    historic = json.loads((ROOT / "audit/sudoku-parity-full-hard-predictions.json").read_text())
    historic = {i: historic[i] for i in selected}
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    report_path = args.output / "report.json"
    if args.resume:
        report = json.loads(report_path.read_text())
        assert report["split_sha256"] == file_hash(split_path)
        assert report["indices"] == selected
        for name, fingerprint in report["source_files"].items():
            assert file_hash(ROOT / name) == fingerprint, name
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(status="running", scope="training_only_no_calibration_or_test_inference",
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      split_sha256=file_hash(split_path), protocol_sha256=file_hash(protocol_path),
                      source_files=source_fingerprints(), indices=selected, n=1000, batch_size=8,
                      eager_control_rule="batch_number_mod_10_equals_0_or_last_batch; cached parity on every batch",
                      batches=[], attempts=[])
    attempt = dict(started_utc=datetime.now(timezone.utc).isoformat())
    report["attempts"].append(attempt)
    report["status"] = "running"
    write_json(report_path, report)
    t0 = time.perf_counter()
    try:
        model = SudokuModel(ROOT / "artifacts/sudoku/hard")
        initial_state = state_hash(model)
        model.predict(model.prefix([puzzles[i] for i in selected[:8]]))
        for batch_number, start in enumerate(range(0, 1000, 8)):
            indices = selected[start:start + 8]
            name = f"batch-{batch_number:03d}.npz"
            destination = args.output / name
            if batch_number < len(report["batches"]):
                old = report["batches"][batch_number]
                assert old["indices"] == indices and old["sha256"] == file_hash(destination)
                continue
            if destination.exists():
                raise RuntimeError("Uncommitted shard exists; preserve it and use a new run directory")
            prefix = model.prefix([puzzles[i] for i in indices])
            reference = model.predict(prefix)
            observed, trace, raw, timing = instrumented_prediction(model, prefix)
            assert_prediction_equal(reference, observed)
            eager_checked = batch_number % 10 == 0 or start + 8 >= 1000
            if eager_checked:
                eager = model.predict(prefix, optimized=False)
                assert_prediction_equal(eager, observed)
                del eager
            if observed.ids.tolist() != [historic[i] for i in indices]:
                raise AssertionError("Historical answer mismatch")
            failures = np.array([int(historic[i] != references[i]) for i in indices], dtype=np.int64)
            np.savez_compressed(destination, sample_indices=np.array(indices), prefix=prefix.numpy(),
                                failure=failures, **trace)
            report["batches"].append(dict(indices=indices, file=name, sha256=file_hash(destination),
                bytes=destination.stat().st_size, failures=int(failures.sum()),
                cached_bitwise_equal=True, eager_bitwise_checked=eager_checked,
                historical_answers_equal=True, **timing))
            del reference, observed, trace, raw
            write_json(report_path, report)
            if batch_number % 10 == 0 or start + 8 >= 1000:
                print(f"training extraction {start + len(indices)}/1000: exact cached parity", flush=True)
        assert sum(row["failures"] for row in report["batches"]) == 137
        assert initial_state == state_hash(model)
        for name, fingerprint in report["source_files"].items():
            assert file_hash(ROOT / name) == fingerprint, name
        attempt["wall_seconds"] = time.perf_counter() - t0
        report.update(status="complete", model_state_sha256_before_and_after=initial_state,
                      failures=137, successes=863, stored_bytes=sum(row["bytes"] for row in report["batches"]),
                      eager_checked_boards=sum(len(row["indices"]) for row in report["batches"] if row["eager_bitwise_checked"]),
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      probe_training="not part of this extraction")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        attempt["wall_seconds"] = time.perf_counter() - t0
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("status", "failures", "stored_bytes", "eager_checked_boards")}), flush=True)


if __name__ == "__main__":
    main()
