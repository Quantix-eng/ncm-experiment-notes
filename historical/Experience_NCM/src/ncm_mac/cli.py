"""Reproducible official Sudoku evaluation on the validated CPU path."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import torch
from ncm_mac.sudoku import SudokuModel, load_boards, metrics, verify_assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, choices=[1, 2, 4], default=1)
    ap.add_argument("--threads", type=int, default=None, help="Threads per worker (default: 4/workers)")
    ap.add_argument("--n", type=int, default=0, help="0 = complete official test set")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.threads = a.threads if a.threads is not None else max(1, 4 // a.workers)
    if a.batch_size <= 0 or a.threads <= 0 or a.n < 0:
        ap.error("batch size and threads must be positive; n must be nonnegative")
    torch.set_num_threads(a.threads)
    verify_assets(a.root, a.difficulty, include_data=True)
    dd = a.root / "data/sudoku" / a.difficulty
    puzzles, solutions = load_boards(dd / "puzzles.pickle"), load_boards(dd / "solutions.pickle")
    if a.n:
        puzzles, solutions = puzzles[:a.n], solutions[:a.n]
    predictions, seconds = [], []
    worker_memory, startup_seconds = {}, 0.0
    checkpoint = a.root / "artifacts/sudoku" / a.difficulty
    pipeline_started = time.perf_counter()
    if a.workers == 1:
        started = time.perf_counter()
        model = SudokuModel(checkpoint)
        model.predict(model.prefix(puzzles[:a.batch_size]))  # warmup, excluded
        startup_seconds = time.perf_counter() - started
        for start in range(0, len(puzzles), a.batch_size):
            prefix = model.prefix(puzzles[start:start + a.batch_size])
            t = time.perf_counter()
            result = model.predict(prefix)
            seconds.append(time.perf_counter() - t)
            predictions.extend(result.ids.tolist())
    else:
        from ncm_mac.parallel import SudokuPool
        started = time.perf_counter()
        with SudokuPool(checkpoint, a.workers, a.threads) as pool:
            list(pool.map([puzzles[:a.batch_size]] * (2 * a.workers)))
            startup_seconds = time.perf_counter() - started
            batches = [puzzles[i:i + a.batch_size] for i in range(0, len(puzzles), a.batch_size)]
            started = time.perf_counter()
            for result in pool.map(batches):
                predictions.extend(result.ids)
                worker_memory[result.worker_pid] = max(worker_memory.get(result.worker_pid, 0), result.peak_rss_bytes)
            seconds.append(time.perf_counter() - started)
    pipeline_seconds = time.perf_counter() - pipeline_started
    raw = json.dumps(predictions, separators=(",", ":")).encode()
    report = dict(difficulty=a.difficulty, backend="cpu", dtype="float32", batch_size=a.batch_size,
                  threads=a.threads, workers=a.workers, startup_seconds=startup_seconds,
                  inference_pipeline_seconds=pipeline_seconds,
                  timing_scope="tensor inference" if a.workers == 1 else "dispatch, inference and IDs IPC",
                  python=platform.python_version(), torch=torch.__version__,
                  macos=platform.mac_ver()[0], arch=platform.machine(),
                  metrics=metrics(predictions, solutions, puzzles),
                  generation_seconds=sum(seconds), boards_per_second=len(puzzles) / sum(seconds),
                  median_batch_seconds=statistics.median(seconds) if a.workers == 1 else None,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  worker_peak_rss_bytes=worker_memory,
                  memory_scope="parent RSS and separate per-worker RSS; shared pages may overlap",
                  predictions_sha256=hashlib.sha256(raw).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    a.output.with_suffix(".predictions.json").write_bytes(raw)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
