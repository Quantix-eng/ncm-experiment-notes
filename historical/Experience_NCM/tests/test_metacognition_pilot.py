"""Scientific boundaries: group leakage, future leakage, grammar, hook cleanup."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/metacognition"))
from design import partitions, pilot_indices
from features import CELL_POS, build_features, observed_counts, prompt_cells
from trace import instrumented_prediction

LENGTHS = [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]


def synthetic_trace():
    rng = np.random.default_rng(7)
    grid = np.array([((r * 3 + r // 3 + c) % 9) + 1 for r in range(9) for c in range(9)])
    ids = np.full((1, 89), 10, dtype=np.int64)
    ids[:, CELL_POS] = grid
    given = ids.copy()
    given[:, CELL_POS[30:]] = 0
    prefix = np.concatenate([np.full((1, 1), 11), given, np.full((1, 1), 11)], axis=1)
    cp = rng.uniform(.1, 1., (1, sum(LENGTHS), 16)); cp /= cp.sum(-1, keepdims=True)
    dp = rng.uniform(.1, 1., (1, 12, 89, 12)); dp /= dp.sum(-1, keepdims=True)
    trace = dict(codes=cp.argmax(-1), code_probs=cp, decoder_probs=dp,
                 decoded_ids=np.repeat(ids[:, None], 12, axis=1),
                 hidden_summary=rng.normal(size=(1, 12, 1024)))
    return prefix, trace


class MetacognitionPilot(unittest.TestCase):
    def test_partition_reproducibility_class_counts_and_pilot_training_only(self):
        labels = [1] * 274 + [0] * 1726
        groups = [str(i) for i in range(2000)]
        p = partitions(labels, groups)
        self.assertEqual(p, partitions(labels, groups))
        self.assertEqual([len(p[k]) for k in ("train", "calibration", "test")], [1000, 500, 500])
        self.assertEqual([sum(labels[i] for i in p[k]) for k in ("train", "calibration", "test")], [137, 69, 68])
        self.assertEqual(len(set(sum(p.values(), []))), 2000)
        pilot = pilot_indices(p["train"], labels)
        self.assertTrue(set(pilot).issubset(p["train"]))
        self.assertEqual(sum(labels[i] for i in pilot), 8)
        with self.assertRaises(ValueError):
            partitions(labels, ["same"] * 2000)

    def test_future_corruption_cannot_affect_level9_predictors(self):
        prefix, trace = synthetic_trace()
        before = build_features(prefix, trace, 9, LENGTHS)
        trace["code_probs"][:, 116:] = np.nan
        trace["codes"][:, 116:] = 99
        trace["decoder_probs"][:, 9:] = np.nan
        trace["decoded_ids"][:, 9:] = 99
        trace["hidden_summary"][:, 9:] = np.nan
        after = build_features(prefix, trace, 9, LENGTHS)
        self.assertTrue(all(np.array_equal(before[k], after[k]) for k in before))
        self.assertEqual(before["rich"].shape, (1, 13877))
        self.assertEqual(before["hidden_raw"].shape, (1, 1024))

    def test_invalid_token_keeps_cell_position_and_is_rejected(self):
        prefix, trace = synthetic_trace()
        given = prompt_cells(prefix)
        ids = trace["decoded_ids"][:, -1].copy()
        self.assertTrue(observed_counts(ids, given)[-1][0])
        ids[:, CELL_POS[40]] = 0
        invalid, _, _, _, valid = observed_counts(ids, given)
        self.assertEqual(invalid[0], 1)
        self.assertFalse(valid[0])
        self.assertEqual(ids[:, CELL_POS].shape[1], 81)
        # A row separator misplaced in a cell must not shift later positions.
        ids[:, CELL_POS[40]] = 10
        self.assertEqual(observed_counts(ids, given)[0][0], 1)

    def test_hooks_removed_even_if_inference_raises(self):
        blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)])
        heads = torch.nn.ModuleList([torch.nn.Linear(512, 16)])
        net = SimpleNamespace(levels=[1], transformer=SimpleNamespace(blocks=blocks), lm_heads=heads)
        def fail(prefix):
            raise RuntimeError("injected test failure")
        model = SimpleNamespace(net=net, vqvae=SimpleNamespace(vq=object()), predict=fail)
        with self.assertRaisesRegex(RuntimeError, "injected test failure"):
            instrumented_prediction(model, torch.zeros((1, 91), dtype=torch.long))
        self.assertEqual(len(blocks[7]._forward_hooks), 0)
        self.assertEqual(len(heads[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
