"""Describe intermediate grids on training examples only; never select features."""
import json

import numpy as np

from features import CELL_POS, observed_counts, prompt_cells
from run_pilot import ROOT, file_hash, load_boards, write_json


def main():
    directory = ROOT / "audit/metacognition/train-v0.2"
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "complete"
    split = json.loads((ROOT / "audit/metacognition/split-v0.2.json").read_text())
    solutions = load_boards(ROOT / "data/sudoku/hard/solutions.pickle")
    solutions = {i: solutions[i] for i in split["partition_indices"]["train"]}
    correct, valid, malformed, regressions = np.zeros(12, int), np.zeros(12, int), np.zeros(12, int), 0
    any_regression, rows = 0, []
    for batch in report["batches"]:
        path = directory / batch["file"]
        assert file_hash(path) == batch["sha256"]
        with np.load(path, allow_pickle=False) as trace:
            indices = trace["sample_indices"].tolist()
            rows.extend(indices)
            gold = np.array([solutions[i] for i in indices])
            ids = trace["decoded_ids"]
            given = prompt_cells(trace["prefix"])
            correct += (ids == gold[:, None, :]).all(2).sum(0)
            counts = (ids[:, :, CELL_POS] == gold[:, None, CELL_POS]).sum(2)
            decreases = np.diff(counts, axis=1) < 0
            regressions += int(decreases.sum())
            any_regression += int(decreases.any(1).sum())
            for k in range(12):
                invalid, bad_sep, _, _, okay = observed_counts(ids[:, k], given)
                valid[k] += int(okay.sum())
                malformed[k] += int(((invalid > 0) | (bad_sep > 0)).sum())
    assert rows == split["partition_indices"]["train"]
    assert correct[-1] == valid[-1] == 863
    output = dict(scope="training_only_descriptive_not_policy_evaluation", n=len(rows),
                  extraction_report_sha256=file_hash(directory / "report.json"),
                  exact_correct_by_level=correct.tolist(), verifier_valid_by_level=valid.tolist(),
                  malformed_by_level=malformed.tolist(), examples_with_cell_accuracy_regression=any_regression,
                  cell_accuracy_regressive_transitions=regressions)
    write_json(directory / "descriptive-evaluation.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
