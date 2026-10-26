"""Synthetic paired-AUC design sensitivity; no NCM data or model execution.

Equal-variance binormal scores with specified within-class correlation.
Paired DeLong placement variance, normal CIs, conditional fixed class counts.
This omits uncertainty from training/calibrating predictors and is not a
power guarantee for an unknown future score distribution.
"""
import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def paired_auc(positive, negative):
    """Return AUC difference (column 1 - 0) and paired asymptotic SE."""
    m, n = len(positive), len(negative)
    vp, vn = np.empty_like(positive), np.empty_like(negative)
    for j in range(2):
        sp, sn = np.sort(positive[:, j]), np.sort(negative[:, j])
        vp[:, j] = (np.searchsorted(sn, positive[:, j], side="left") +
                    np.searchsorted(sn, positive[:, j], side="right")) / (2 * n)
        vn[:, j] = 1 - (np.searchsorted(sp, negative[:, j], side="left") +
                        np.searchsorted(sp, negative[:, j], side="right")) / (2 * m)
    delta = float(vp[:, 1].mean() - vp[:, 0].mean())
    variance = np.var(vp[:, 1] - vp[:, 0], ddof=1) / m + np.var(vn[:, 1] - vn[:, 0], ddof=1) / n
    return delta, sqrt(max(0.0, float(variance)))


def check_placements():
    p = np.array([[1., 2.], [2., 1.], [3., 4.]])
    n = np.array([[0., 1.], [2., 0.], [4., 3.], [1., 2.]])
    brute = ((p[:, None, :] > n[None, :, :]).mean(axis=(0, 1)) +
             0.5 * (p[:, None, :] == n[None, :, :]).mean(axis=(0, 1)))
    delta, _ = paired_auc(p, n)
    assert np.isclose(delta, brute[1] - brute[0])
    d, se = paired_auc(np.repeat(p[:, :1], 2, axis=1), np.repeat(n[:, :1], 2, axis=1))
    assert d == 0 and se == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=ROOT / "audit/metacognition/design-power-sensitivity.json")
    args = ap.parse_args()
    if args.replicates < 100:
        ap.error("Use at least 100 replicates")
    check_placements()
    rows = []
    normal = NormalDist()
    for size in (500, 1000, 2000, 5000):
        failures = round(size * .137)
        successes = size - failures
        for auc0 in (.70, .85):
            for correlation in (.3, .7, .95):
                for effect in (0., .03, .06):
                    seed = [20261023, size, round(auc0 * 100), round(correlation * 100), round(effect * 100)]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    means = sqrt(2) * np.array([normal.inv_cdf(auc0), normal.inv_cdf(auc0 + effect)])
                    estimates, ses = [], []
                    for _ in range(args.replicates):
                        zp = rng.standard_normal((failures, 2))
                        zn = rng.standard_normal((successes, 2))
                        for z in (zp, zn):
                            z[:, 1] = correlation * z[:, 0] + sqrt(1 - correlation ** 2) * z[:, 1]
                        d, se = paired_auc(zp + means, zn)
                        estimates.append(d)
                        ses.append(se)
                    estimates, ses = np.array(estimates), np.array(ses)
                    lower, upper = estimates - 1.959963984540054 * ses, estimates + 1.959963984540054 * ses
                    detected = lower > 0
                    p = float(detected.mean())
                    rows.append(dict(n=size, failures=failures, successes=successes,
                        auc_baseline=auc0, true_delta=effect, within_class_score_correlation=correlation,
                        mean_delta=float(estimates.mean()), mean_ci_width=float((upper - lower).mean()),
                        ci_coverage=float(((lower <= effect) & (upper >= effect)).mean()),
                        positive_gain_detection_probability=p,
                        detection_monte_carlo_se=sqrt(p * (1 - p) / args.replicates),
                        two_sided_null_rejection_probability=float(((lower > 0) | (upper < 0)).mean()),
                        old_promising_rule_probability=float((detected & (estimates >= .03)).mean())))
    report = dict(date="2026-10-23", kind="synthetic_design_sensitivity_not_NCM_results",
        method="equal-variance paired binormal scores, fixed class counts, DeLong placement SE and normal 95% CI",
        estimand="conditional test-score AUC difference; positive class = model failure",
        limitations=["No probe fitting or calibration simulated", "Unknown actual AUC and within-class score correlation",
                     "Normal CI approximation, not bootstrap", "Failure prevalence on a new generator is unknown"],
        replicates=args.replicates, numpy=np.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sanity_checks="AUC matches pairwise brute force including ties; identical scores have zero difference and SE",
        scenarios=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), scenarios=len(rows), replicates=args.replicates)))
    for size in (500, 1000, 2000, 5000):
        subset = [r for r in rows if r["n"] == size and r["true_delta"] == .03]
        print(size, "positive_gain_detection_range", min(r["positive_gain_detection_probability"] for r in subset),
              max(r["positive_gain_detection_probability"] for r in subset))


if __name__ == "__main__":
    main()
