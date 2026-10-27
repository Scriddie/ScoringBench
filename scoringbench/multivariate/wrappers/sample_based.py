"""Sample-based multivariate wrapper base class.

``SampleBasedWrapper`` is the multivariate analogue of the univariate class of
the same name, but it never touches a grid or PMF: it accumulates ``(n_test, m,
d)`` draws (topping up until R̂ converges or the sample cap is hit) and wraps
them in a :class:`MultivariateSamplePrediction`.

Subclasses only implement :meth:`_draw_samples(X, n) -> (n_test, n, d)`.
"""

from __future__ import annotations

import warnings

import numpy as np

from .. import config as _config
from ..config import N_DRAWS
from ..convergence import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_N_GROUPS,
    assess_sample_prediction,
    log_sample_cap_hit,
)
from ..prediction import MultivariateSamplePrediction
from .base import MultivariateWrapper


class SampleBasedWrapper(MultivariateWrapper):
    """Base for models whose predictive law is accessed by sampling.

    Class attributes
    ----------------
    N_SAMPLES : int
        Target number of draws per test instance (defaults to the benchmark-wide
        ``config.N_DRAWS`` so every model emits the same ``m`` — required for a
        fair leaderboard, see ``config.py``).
    SAMPLE_CHUNK : int
        Draws requested per call to :meth:`_draw_samples`.  Set equal to
        ``N_SAMPLES`` for one-shot samplers.
    """

    N_SAMPLES: int = int(N_DRAWS)
    SAMPLE_CHUNK: int = int(N_DRAWS)

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        """Return an ``(n_test, n_samples, d)`` array of conditional draws."""
        raise NotImplementedError

    def _collect_samples(self, X) -> np.ndarray:
        """Accumulate draws in chunks -> (n_test, m, d).

        After the initial ``N_SAMPLES`` draws, if ``config.CONV_ENABLED`` is
        set the multivariate R̂ (Vats & Knudson 2020, det-form) is evaluated on
        a subsample of test instances.  If R̂ has not converged (max over
        instances >= ``DEFAULT_EPSILON``), additional draws are collected in
        ``DEFAULT_BATCH_SIZE`` chunks until convergence or ``DEFAULT_MAX_SAMPLES``
        is reached.  There is no wall-clock budget.  If the cap is hit without
        converging it is recorded via :func:`log_sample_cap_hit`.  No draws are
        discarded (i.i.d. MC, not MCMC).
        """
        target = int(self.N_SAMPLES)
        chunk = int(self.SAMPLE_CHUNK) or target
        collected: list[np.ndarray] = []
        n_have = 0
        while n_have < target:
            take = min(chunk, target - n_have)
            s = np.asarray(self._draw_samples(X, take), dtype=np.float64)
            if s.ndim == 2:
                # (n_test, d) single-draw -> promote to (n_test, 1, d)
                s = s[:, None, :]
            if s.ndim != 3:
                raise ValueError(
                    f"_draw_samples must return (n_test, n, d); got shape {s.shape}"
                )
            collected.append(s)
            n_have += s.shape[1]
        if not collected:
            raise RuntimeError("No samples were drawn.")
        samples = np.concatenate(collected, axis=1)

        # --- adaptive top-up: draw more until R̂ < DEFAULT_EPSILON ------------
        if not getattr(_config, "CONV_ENABLED", False):
            return samples

        last_rhat: float | None = None
        while samples.shape[1] < DEFAULT_MAX_SAMPLES:
            pred_tmp = MultivariateSamplePrediction(samples=samples)
            try:
                diag = assess_sample_prediction(
                    pred_tmp,
                    n_groups=DEFAULT_N_GROUPS,
                    epsilon=DEFAULT_EPSILON,
                    max_instances=DEFAULT_MAX_INSTANCES,
                )
            except ValueError as exc:
                warnings.warn(
                    f"convergence check skipped (adaptive top-up): {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                break
            last_rhat = diag.rhat_max
            if diag.rhat_max < DEFAULT_EPSILON:
                break
            extra = np.asarray(self._draw_samples(X, DEFAULT_BATCH_SIZE), dtype=np.float64)
            if extra.ndim == 2:
                extra = extra[:, None, :]
            samples = np.concatenate([samples, extra], axis=1)
        else:
            # Loop ended because the sample cap was reached, not via a
            # convergence/exception break -> ensemble did not converge.
            log_sample_cap_hit(
                model=type(self).__name__,
                n_samples=int(samples.shape[1]),
                rhat_max=last_rhat,
            )

        return samples

    def predict_ensemble(self, X) -> MultivariateSamplePrediction:
        samples = self._collect_samples(X)
        return MultivariateSamplePrediction(samples=samples)

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)
