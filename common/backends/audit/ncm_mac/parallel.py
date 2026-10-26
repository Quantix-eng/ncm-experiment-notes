"""Ordered, persistent CPU workers; original batch boundaries stay unchanged."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import multiprocessing
import os
import resource
import sys

import torch
from ncm_mac.sudoku import SudokuModel

_MODEL = None


@dataclass
class BatchResult:
    ids: list
    worker_pid: int
    peak_rss_bytes: int
    fingerprints: list | None = None


def prediction_fingerprints(prediction):
    return [hashlib.sha256(f"{x.dtype}:{tuple(x.shape)}:".encode() + x.contiguous().numpy().tobytes()).hexdigest()
            for x in (prediction.codes, prediction.reconstruction, prediction.logits)]


def _initialize(checkpoint_dir, threads):
    global _MODEL
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _MODEL = SudokuModel(checkpoint_dir)


def _predict(task):
    boards, audit = task
    prediction = _MODEL.predict(_MODEL.prefix(boards))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return BatchResult(prediction.ids.tolist(), os.getpid(), rss,
                       prediction_fingerprints(prediction) if audit else None)


class SudokuPool:
    """One checkpoint per pool; callers provide complete batches in input order."""
    def __init__(self, checkpoint_dir, workers=4, threads=1):
        if workers not in (1, 2, 4) or threads < 1:
            raise ValueError("Use 1, 2 or 4 workers and a positive thread count")
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=_initialize,
            initargs=(str(checkpoint_dir), threads))

    def map(self, batches, *, audit=False):
        # executor.map preserves ordering and sends each batch to one worker.
        return self.executor.map(_predict, ((batch, audit) for batch in batches), chunksize=1)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
