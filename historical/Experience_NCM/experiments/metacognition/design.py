"""Deterministic, disjoint design for the exploratory hard-Sudoku corpus."""
import hashlib
import json

import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def partitions(failures, solution_groups, seed=20261023):
    y = np.asarray(failures, dtype=np.int64)
    if len(y) != len(solution_groups) or len(set(solution_groups)) != len(y):
        raise ValueError("This initial design requires one hard puzzle per unique solution")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Need both outcome classes")
    n = len(y)
    ntrain, ncal = n // 2, n // 4
    nf = int(y.sum())
    ftrain, fcal = int(np.floor(nf * .5 + .5)), int(np.floor(nf * .25 + .5))
    rng = np.random.default_rng(seed)
    pos, neg = rng.permutation(np.flatnonzero(y)), rng.permutation(np.flatnonzero(y == 0))
    ptrain, pcal = ntrain - ftrain, ncal - fcal
    result = {}
    for name, a, b in (
        ("train", pos[:ftrain], neg[:ptrain]),
        ("calibration", pos[ftrain:ftrain + fcal], neg[ptrain:ptrain + pcal]),
        ("test", pos[ftrain + fcal:], neg[ptrain + pcal:])):
        result[name] = sorted(np.concatenate([a, b]).tolist())
    flat = sum(result.values(), [])
    if len(flat) != n or len(set(flat)) != n or set(flat) != set(range(n)):
        raise AssertionError("Partition overlap or incomplete coverage")
    return result


def pilot_indices(train_indices, failures, seed=20261028):
    rng = np.random.default_rng(seed)
    good = [i for i in train_indices if not failures[i]]
    bad = [i for i in train_indices if failures[i]]
    if len(good) < 24 or len(bad) < 8:
        raise ValueError("Need 24 successes and 8 failures within training")
    return rng.permutation(np.concatenate([rng.choice(good, 24, replace=False),
                                          rng.choice(bad, 8, replace=False)])).tolist()


def cv_folds(train_indices, failures, seed=20261026):
    rng = np.random.default_rng(seed)
    result = {}
    for label in (0, 1):
        candidates = rng.permutation([i for i in train_indices if failures[i] == label])
        for j, index in enumerate(candidates):
            result[str(int(index))] = j % 5
    return result
