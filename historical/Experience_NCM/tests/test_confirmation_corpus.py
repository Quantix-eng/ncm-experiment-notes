import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/metacognition"))
from confirmation_corpus import backtracking, exact_cover_count, known_transform_key


class ConfirmationSolvers(unittest.TestCase):
    def test_solvers_agree_on_unique_impossible_and_multiple(self):
        board = [int(c) for c in "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
        solution = backtracking(board)[0]
        self.assertEqual(len(backtracking(board)), 1)
        self.assertEqual(exact_cover_count(board), 1)
        self.assertEqual(exact_cover_count(solution), 1)
        invalid = board.copy()
        invalid[1] = 5
        self.assertEqual(backtracking(invalid), [])
        self.assertEqual(exact_cover_count(invalid), 0)
        self.assertEqual(len(backtracking([0] * 81)), 2)
        self.assertEqual(exact_cover_count([0] * 81), 2)

    def test_known_transforms_share_key(self):
        solution = backtracking([0] * 81, limit=1)[0]
        transformed = [10 - solution[(8 - c) * 9 + r] for r in range(9) for c in range(9)]
        self.assertEqual(known_transform_key(solution), known_transform_key(transformed))


if __name__ == "__main__":
    unittest.main()
