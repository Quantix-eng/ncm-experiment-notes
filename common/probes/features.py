"""Gold-free features, bounded to the requested generation level.

All arrays have a batch dimension. Expected Sudoku positions are never filtered
according to predicted token values; malformed tokens retain their position.
"""
import numpy as np

CELL_POS = np.array([i for i in range(89) if i % 10 != 9])
SEP_POS = np.array([i for i in range(89) if i % 10 == 9])
GROUPS = ([list(range(r * 9, r * 9 + 9)) for r in range(9)] +
          [list(range(c, 81, 9)) for c in range(9)] +
          [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)]
           for br in (0, 3, 6) for bc in (0, 3, 6)])


def prompt_cells(prefix):
    prefix = np.asarray(prefix)
    if prefix.ndim != 2 or prefix.shape[1] != 91 or not np.all(prefix[:, [0, -1]] == 11):
        raise ValueError("Expected 91-token BOS-delimited prompts")
    if not np.all(prefix[:, 1:-1][:, SEP_POS] == 10):
        raise ValueError("Invalid prompt row separators")
    cells = prefix[:, 1:-1][:, CELL_POS]
    if np.any((cells < 0) | (cells > 9)) or not np.all((cells != 0).sum(1) == 30):
        raise ValueError("Expected exactly 30 clues for hard")
    return cells


def observed_counts(ids, given):
    cells = ids[:, CELL_POS]
    invalid = ((cells < 1) | (cells > 9)).sum(1)
    bad_separators = (ids[:, SEP_POS] != 10).sum(1)
    clue_errors = ((given != 0) & (cells != given)).sum(1)
    pairs = np.zeros((len(ids), 3), dtype=np.int64)
    for j, group in enumerate(GROUPS):
        for value in range(1, 10):
            count = (cells[:, group] == value).sum(1)
            pairs[:, j // 9] += count * (count - 1) // 2
    valid = (invalid == 0) & (bad_separators == 0) & (clue_errors == 0) & (pairs.sum(1) == 0)
    return invalid, bad_separators, clue_errors, pairs, valid


def scalar_summary(values):
    return np.column_stack([values.mean(1), values.std(1, ddof=0),
                            np.quantile(values, [.1, .5, .9], axis=1, method="linear").T])


def probability_features(probabilities, denominator):
    order = np.sort(probabilities, axis=-1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(-1) / np.log(denominator)
    return entropy, order[..., -1] - order[..., -2], order[..., -1]


def build_features(prefix, trace, level, lengths):
    if not 1 <= level <= len(lengths):
        raise ValueError("Level out of range")
    given = prompt_cells(prefix)
    b = len(given)
    blank, clues = given == 0, given != 0
    puzzle = [np.eye(10, dtype=np.float64)[given].reshape(b, -1)]
    puzzle.append(np.column_stack([(given[:, group] != 0).sum(1) / 9 for group in GROUPS]))
    puzzle.append(np.column_stack([(given == v).sum(1) / 30 for v in range(1, 10)]))
    puzzle = np.concatenate(puzzle, axis=1)
    summaries, offset, previous = [], 0, None
    for k, length in enumerate(lengths[:level]):
        cp = trace["code_probs"][:, offset:offset + length]
        selected = trace["codes"][:, offset:offset + length]
        dec = trace["decoder_probs"][:, k]
        ids = trace["decoded_ids"][:, k]
        dpcells = dec[:, CELL_POS]
        free_probs = dpcells[blank].reshape(b, 51, 12)
        values = list(probability_features(cp, 16))
        chosen = np.take_along_axis(cp, selected[..., None], axis=-1)[..., 0]
        values.append(np.log(np.maximum(chosen, 1e-12)))
        values.extend(probability_features(free_probs, 12))
        actual_clue_prob = np.take_along_axis(dpcells, given[..., None], axis=-1)[..., 0]
        values.extend([actual_clue_prob[clues].reshape(b, 30), dec[:, SEP_POS, 10]])
        parts = [scalar_summary(v) for v in values]
        invalid, bad_sep, clue_bad, pairs, valid = observed_counts(ids, given)
        parts.append(np.column_stack([invalid / 81, bad_sep / 8, clue_bad / 30,
                                      pairs / 324, valid.astype(float)]))
        if previous is None:
            parts.append(np.zeros((b, 3)))
        else:
            changes = ids[:, CELL_POS] != previous[:, CELL_POS]
            parts.append(np.column_stack([(changes & blank).sum(1) / 51,
                                          (changes & clues).sum(1) / 30, np.ones(b)]))
        summaries.append(np.concatenate(parts, axis=1))
        previous = ids
        offset += length
    summary = np.concatenate([puzzle, *summaries], axis=1)
    history = np.eye(12, dtype=np.float64)[trace["decoded_ids"][:, :level]].reshape(b, -1)
    rich = np.concatenate([summary, trace["decoder_probs"][:, level - 1].reshape(b, -1),
                           trace["code_probs"][:, :offset].reshape(b, -1), history], axis=1)
    hidden = trace["hidden_summary"][:, level - 1].copy()
    output = dict(puzzle=puzzle, summary=summary, rich=rich, hidden_raw=hidden)
    if not all(np.isfinite(v).all() for v in output.values()):
        raise ValueError("Nonfinite feature (or access to poisoned future states)")
    return output
