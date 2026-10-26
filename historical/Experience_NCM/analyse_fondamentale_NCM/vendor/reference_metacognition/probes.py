"""Fold-local preprocessing and the explicitly specified ridge logistic probe."""
import numpy as np
import torch
import torch.nn.functional as F


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(0)
        sd = x.std(0, ddof=0)
        self.keep = sd > 0
        self.scale = sd[self.keep]
        return self

    def transform(self, x):
        result = (x[:, self.keep] - self.mean[self.keep]) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized predictors")
        return result


class HiddenPCA:
    def fit(self, x, components=16):
        self.standardizer = Standardizer().fit(x)
        z = torch.from_numpy(self.standardizer.transform(x))
        _, _, vt = torch.linalg.svd(z, full_matrices=False)
        self.axes = vt[:components].numpy().copy()
        if len(self.axes) != components:
            raise ValueError("Insufficient rank/dimensions for specified PCA")
        for axis in self.axes:
            if axis[np.argmax(np.abs(axis))] < 0:
                axis *= -1
        scores = z.numpy() @ self.axes.T
        self.score_scaler = Standardizer().fit(scores)
        if not self.score_scaler.keep.all():
            raise ValueError("Degenerate PCA coordinates")
        return self

    def transform(self, x):
        return self.score_scaler.transform(self.standardizer.transform(x) @ self.axes.T)


def fit_logistic(x, y, strength, settings):
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    if not 0 < y.mean() < 1:
        raise ValueError("Both classes required")
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(torch.logit(y.mean())), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], **settings)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + .5 * strength * w.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite logistic objective")
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure()
    state = optimizer.state[w]
    iterations = int(state.get("n_iter", 0))
    gradient = float(torch.cat([w.grad.flatten(), b.grad.reshape(1)]).abs().max())
    if iterations >= settings["max_iter"]:
        raise RuntimeError("LBFGS reached preset iteration cap; no automatic retuning")
    return dict(weight=w.detach().numpy().copy(), intercept=float(b.detach()),
                iterations=iterations, max_abs_gradient=gradient, objective=float(loss.detach()),
                stopping="LBFGS stopped before iteration cap under configured gradient/step/objective criteria")


def predict(fit, x):
    return x @ fit["weight"] + fit["intercept"]


def log_loss(y, score):
    return float(np.mean(np.logaddexp(0., score) - np.asarray(y) * score))


def auc(y, score):
    p = np.asarray(score)[np.asarray(y) == 1]
    n = np.sort(np.asarray(score)[np.asarray(y) == 0])
    if not len(p) or not len(n):
        raise ValueError("Both classes required for AUROC")
    return float(np.mean((np.searchsorted(n, p, side="left") +
                          np.searchsorted(n, p, side="right")) / (2 * len(n))))


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    p = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = np.cumsum(sorted_y)[ends]
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    return dict(n=len(y), failures=int(y.sum()), auc=auc(y, score),
                failure_average_precision=ap, uncalibrated_brier=float(np.mean((p - y) ** 2)),
                log_loss=log_loss(y, score))


def stratified_folds(y, seed, count=5):
    rng = np.random.default_rng(seed)
    result = np.empty(len(y), dtype=int)
    for label in (0, 1):
        ix = rng.permutation(np.flatnonzero(np.asarray(y) == label))
        result[ix] = np.arange(len(ix)) % count
    return result
