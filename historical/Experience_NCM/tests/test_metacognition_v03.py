"""Regression against the observed amplification failure, with held-out spikes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/metacognition"))
from study_v03 import ObservableScaler
from study_v03 import prepared_pair
from final_v03 import transformed
from stability_v03 import inner_assignments


class StablePreprocessing(unittest.TestCase):
    def test_rare_probability_spike_has_bounded_amplification(self):
        train = np.column_stack([np.linspace(1e-21, 1e-14, 40), np.full(40, .03), np.arange(40)])
        scaler = ObservableScaler().fit(train)
        validation = np.array([[6.61e-6, .04, 41.]])
        self.assertEqual(scaler.keep.tolist(), [True, False, True])
        self.assertLess(abs(scaler.transform(validation)[0, 0]), 1e-4)
        self.assertTrue((scaler.scale >= .1).all())
        mean = scaler.mean.copy()
        scaler.transform(np.array([[1., .09, 1e8]]))
        self.assertTrue(np.array_equal(mean, scaler.mean))

    def test_final_transform_replays_fold_local_preprocessing(self):
        rng = np.random.default_rng(19)
        data = {name: rng.normal(size=(48, size)) for name, size in
                (("rich", 22), ("summary", 5), ("hidden_raw", 20))}
        data["indices"] = np.arange(48)
        noise = np.random.default_rng(20261024).standard_normal((2000, 16))[:48]
        train, valid = np.arange(40), np.arange(40, 48)
        names = ["rich", "summary", "rich_hidden_primary", "rich_noise_control"]
        selected = {k: v[valid] for k, v in data.items()}
        for name, (_, expected, saved) in prepared_pair(data, train, valid, names, noise, 2).items():
            self.assertTrue(np.allclose(transformed(saved, selected, name), expected))

    def test_bootstrap_inner_folds_are_nonempty_and_keep_duplicates_together(self):
        y = np.r_[np.zeros(100, dtype=int), np.ones(30, dtype=int)]
        pool = np.r_[np.arange(20, 100), np.arange(105, 130)]
        sampled = np.repeat(pool, 2)
        cv = inner_assignments(y, pool, sampled)
        self.assertEqual(set(cv), set(range(5)))
        for k in range(5):
            self.assertFalse(set(sampled[cv == k]) & set(sampled[cv != k]))


if __name__ == "__main__":
    unittest.main()
