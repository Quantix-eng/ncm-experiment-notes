"""Numeric correctness and fold-local behavior of the exploratory probes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/metacognition"))
from probes import HiddenPCA, Standardizer, auc, fit_logistic, metrics, stratified_folds


class MetacognitionProbes(unittest.TestCase):
    def test_auc_and_average_precision_handle_ties_without_order_dependence(self):
        y = np.array([0, 1, 1, 0])
        score = np.zeros(4)
        self.assertEqual(auc(y, score), .5)
        self.assertEqual(metrics(y, score)["failure_average_precision"], .5)
        self.assertEqual(auc(y, y), 1.)
        self.assertEqual(auc(y, -y), 0.)

    def test_ridge_logistic_matches_scalar_optimum(self):
        torch.set_num_threads(2)
        settings = dict(lr=1., max_iter=500, tolerance_grad=1e-7,
                        tolerance_change=1e-9, line_search_fn="strong_wolfe")
        fitted = fit_logistic(np.array([[-1.], [1.]]), np.array([0, 1]), 1., settings)
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo + hi) / 2
            if mid - 1 / (1 + np.exp(mid)) > 0:
                hi = mid
            else:
                lo = mid
        self.assertAlmostEqual(fitted["weight"][0], (lo + hi) / 2, places=5)
        self.assertAlmostEqual(fitted["intercept"], 0., places=7)

    def test_scaling_and_pca_do_not_fit_on_validation(self):
        rng = np.random.default_rng(6)
        train = rng.normal(size=(32, 20))
        train[:, 0] = 5.
        valid = rng.normal(size=(4, 20)) + 1000
        scaler = Standardizer().fit(train)
        mean_before = scaler.mean.copy()
        self.assertFalse(scaler.keep[0])
        scaler.transform(valid)
        self.assertTrue(np.array_equal(scaler.mean, mean_before))
        pca = HiddenPCA().fit(train)
        axes_before = pca.axes.copy()
        self.assertEqual(pca.transform(valid).shape, (4, 16))
        self.assertTrue(np.array_equal(pca.axes, axes_before))
        self.assertTrue(np.allclose(pca.transform(train).std(0), 1.))

    def test_stratified_folds_reproducible_and_cover_each_example_once(self):
        y = np.array([1] * 137 + [0] * 863)
        fold = stratified_folds(y, 20261026)
        self.assertTrue(np.array_equal(fold, stratified_folds(y, 20261026)))
        for k in range(5):
            self.assertTrue(0 < y[fold == k].sum() < (fold == k).sum())
            self.assertFalse(set(np.flatnonzero(fold == k)) & set(np.flatnonzero(fold != k)))

    def test_prepared_pair_saved_transforms_reproduce_hidden_and_noise(self):
        from train_probes import prepared_pair
        rng = np.random.default_rng(78)
        data = {name: rng.normal(size=(44, columns)) for name, columns in
                (("rich", 30), ("puzzle", 5), ("summary", 8), ("hidden_raw", 20))}
        noise = rng.normal(size=(44, 16))
        train, valid = np.arange(36), np.arange(36, 44)
        names = ["puzzle", "summary", "rich", "rich_hidden_primary", "rich_noise_control", "hidden_shuffle_control"]
        prepared = prepared_pair(data, train, valid, names, noise, 42)
        for name, (a, b, saved) in prepared.items():
            family = name if name in ("puzzle", "summary") else "rich"
            base = data[family][valid]
            replay = (base[:, saved["base_keep"]] - saved["base_mean"][saved["base_keep"]]) / saved["base_scale"]
            if "pca_axes" in saved:
                rows = saved["hidden_validation_source_rows"]
                self.assertEqual(sorted(rows.tolist()), valid.tolist())
                z = data["hidden_raw"][rows]
                z = (z[:, saved["hidden_keep"]] - saved["hidden_mean"][saved["hidden_keep"]]) / saved["hidden_scale"]
                extra = (z @ saved["pca_axes"].T - saved["pca_score_mean"]) / saved["pca_score_scale"]
                replay = np.column_stack([replay, extra])
            elif "noise_mean" in saved:
                extra = (noise[valid][:, saved["noise_keep"]] - saved["noise_mean"][saved["noise_keep"]]) / saved["noise_scale"]
                replay = np.column_stack([replay, extra])
            self.assertTrue(np.allclose(replay, b))
            self.assertEqual(a.shape[1], b.shape[1])


if __name__ == "__main__":
    unittest.main()
