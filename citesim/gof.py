"""Goodness-of-fit tests for citation tails, following the plan in ``email.md``.

1. Anderson-Darling against a *fully specified* power law. ``alpha_0`` and ``x_min`` are
   estimated once from the empirical data and treated as true.
2. Likelihood-ratio test of ``alpha = alpha_0`` against a freely estimated ``alpha``.
3. Vuong's test of power law against log-normal (non-nested).

Citation counts are integers with many ties, so everything uses the discrete power law
``P(x) = x^-alpha / zeta(alpha, x_min)`` for ``x >= x_min``. The A-D statistic is the
discrete version (Choulakian, Lockhart & Stephens 1994), with a Monte Carlo null. A plain
continuous A-D gives infinite statistics or a mis-calibrated level on this data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


def dpl_loglik(tail: np.ndarray, alpha: float, xmin: int) -> float:
    return float(-alpha * np.log(tail).sum() - len(tail) * np.log(special.zeta(alpha, xmin)))


def dpl_mle(tail: np.ndarray, xmin: int, bounds=(1.01, 10.0)) -> tuple[float, float]:
    """Discrete power-law MLE of alpha with its standard error (observed information)."""
    res = optimize.minimize_scalar(lambda a: -dpl_loglik(tail, a, xmin), bounds=bounds, method="bounded",
                                   options={"xatol": 1e-6})
    a, h = res.x, 1e-4
    d2 = (dpl_loglik(tail, a + h, xmin) - 2 * dpl_loglik(tail, a, xmin) + dpl_loglik(tail, a - h, xmin)) / h**2
    return float(a), float(1 / np.sqrt(-d2)) if d2 < 0 else float("nan")


@dataclass
class DiscretePowerLaw:
    """Power law on ``x_min..cap``. The probability beyond ``cap`` is lumped into the last point."""
    alpha: float
    xmin: int
    cap: int = 200_000

    def __post_init__(self):
        x = np.arange(self.xmin, self.cap + 1, dtype=np.float64)
        pmf = np.exp(-self.alpha * np.log(x) - np.log(special.zeta(self.alpha, self.xmin)))
        pmf[-1] += max(0.0, 1.0 - pmf.sum())
        self.pmf = pmf
        self.cdf = np.minimum(np.cumsum(pmf), 1.0)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.xmin + np.searchsorted(self.cdf, rng.random(n), side="left")

    def ad_statistic(self, tail: np.ndarray) -> float:
        """Discrete Anderson-Darling A²: n · Σ (F_n − F)² p / (F (1 − F))."""
        n = len(tail)
        idx = np.clip(tail, self.xmin, self.cap) - self.xmin
        Fn = np.cumsum(np.bincount(idx, minlength=len(self.pmf))) / n
        F = self.cdf
        ok = (F > 0) & (F < 1 - 1e-12)
        return float(n * np.sum((Fn[ok] - F[ok]) ** 2 * self.pmf[ok] / (F[ok] * (1 - F[ok]))))


def ad_test(tail: np.ndarray, model: DiscretePowerLaw, n_mc: int = 999, seed: int = 0) -> tuple[float, float]:
    """A² and its Monte Carlo p-value under the fully specified ``model``."""
    rng = np.random.default_rng(seed)
    obs = model.ad_statistic(tail)
    null = np.array([model.ad_statistic(model.sample(len(tail), rng)) for _ in range(n_mc)])
    return obs, float((1 + np.sum(null >= obs)) / (n_mc + 1))


def lr_test(tail: np.ndarray, alpha0: float, xmin: int) -> dict:
    """LR test of alpha = alpha0 against the free MLE; LR is chi²(1) under H0."""
    a_hat, se = dpl_mle(tail, xmin)
    lr = 2 * (dpl_loglik(tail, a_hat, xmin) - dpl_loglik(tail, alpha0, xmin))
    return {"alpha_hat": a_hat, "alpha_se": se, "LR": float(max(lr, 0.0)), "p_LR": float(stats.chi2.sf(max(lr, 0.0), 1))}


def vuong_test(data: np.ndarray, xmin: int) -> dict:
    """Vuong normalised LR, power law vs log-normal, on the tail above a *fixed* x_min.

    R < 0 favours the log-normal; p is two-sided.
    """
    import warnings

    import powerlaw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(data[data > 0], xmin=xmin, discrete=True, verbose=False,
                           parameter_ranges={"alpha": [1, 10]})
        R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    return {"vuong_R": float(R), "p_vuong": float(p)}


def two_sample_ad(a: np.ndarray, b: np.ndarray, n_resamples: int = 999, seed: int = 0) -> tuple[float, float]:
    """k-sample A-D (simulated tail vs empirical tail), with a permutation p-value."""
    res = stats.anderson_ksamp([a, b], method=stats.PermutationMethod(n_resamples=n_resamples,
                                                                       random_state=seed))
    return float(res.statistic), float(res.pvalue)
