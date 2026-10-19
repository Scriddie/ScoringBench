"""condensité (LightGBM regressor) wrapper for ScoringBench.

condensité (https://github.com/Scriddie/condensite) turns conditional density
estimation into a single nonparametric regression task. With ``ỹ`` the
min-max-scaled target and ``M`` auxiliary points ``u ~ U(0, 1)`` per
observation, a regressor is fitted on ``((x_i, u_im), K_h(ỹ_i - u_im))``. Note
that condensité encodes an approach that can be used to wrap any regressor; here
we use LightGBM gradient-boosted trees. The regression function is the
kernel-smoothed conditional density ``(p(· | x) * K_h)(u)``. ``predict``
evaluates it on a shared grid spanning the training range of ``y``, which maps
directly onto the ScoringBench shared-grid PMF via
:func:`grid_density_to_distribution`.

Paper: A. G. Reisach, O. Collier, A. Luedtke, A. Chambaz (2025), "Transforming
Conditional Density Estimation Into a Single Nonparametric Regression Task",
https://arxiv.org/abs/2511.18530. License: BSD-3-Clause.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import DistributionPrediction, ProbabilisticWrapper
from .density_based import grid_density_to_distribution

# Canonical default hyperparameters. Single source of truth shared by the
# benchmark's MODELS registration and the integration tests. M*h = 1 and
# n_grid = 2/h keep the auxiliary sampling and the evaluation grid matched to h;
# min_leaf_mass and u_max_bin are explained in CondensiteLGBMWrapper.fit.
CONDENSITE_LGBM_PRESET = dict(
    h=0.005,
    M=200,
    min_leaf_mass=0.2,
    n_grid=400,
    u_max_bin=1023,
    regression_params={
        "num_leaves": 100,
        "max_bin": 255,
        "num_iterations": 100,
        "learning_rate": 0.1,
    },
)


class _LGBMPredictor:
    """LightGBM booster as a condensité predictor.

    Duck-types ``condensite.CondensitePredictor`` (``Condensite`` only calls
    ``device``/``to``/``fit``/``predict``), which keeps the condensité import
    lazy.
    """

    def __init__(self, params: dict):
        self.params = params
        self.booster = None

    def device(self):
        return "cpu"

    def to(self, device):
        return self

    def forward(self, features):
        return self.booster.predict(features.numpy())

    def predict(self, features):
        return torch.as_tensor(self.forward(features), dtype=torch.float32)

    def fit(self, dataset):
        import lightgbm as lgb

        # Assemble the n*M expanded rows by indexing; the per-item DataLoader
        # path does the same thing ~1000x slower.
        features = torch.cat([dataset.x[dataset.j_map], dataset.y_aux.reshape(-1, 1)], dim=1)
        params = dict(self.params)
        num_boost_round = params.pop("num_iterations")
        train = lgb.Dataset(features.numpy(), label=dataset.targets.reshape(-1).numpy(),
                            params={"max_bin_by_feature": params.pop("max_bin_by_feature")})
        self.booster = lgb.train(params, train, num_boost_round=num_boost_round)
        return None  # no train MSE: it would cost a full predict on n*M rows


class CondensiteLGBMWrapper(ProbabilisticWrapper):
    """condensité CDE with a LightGBM regressor and a grid-derived PMF.

    Parameters
    ----------
    h : float, optional
        Kernel bandwidth on the min-max-scaled target, i.e. ``h * (y_max - y_min)``
        in original units. Defaults to the :data:`CONDENSITE_LGBM_PRESET` value.
    M : int, optional
        Number of auxiliary points ``u`` per training observation. Defaults to
        the :data:`CONDENSITE_LGBM_PRESET` value.
    min_leaf_mass : float, optional
        Lower bound on ``k * w`` for a leaf spanning ``k`` observations and a
        ``u``-interval of width ``w``; sets ``min_data_in_leaf = min_leaf_mass * M``
        (see :meth:`fit`). Defaults to the :data:`CONDENSITE_LGBM_PRESET` value.
    n_grid : int, optional
        Number of points on the shared ``y``-grid on which the density is
        evaluated. Defaults to the :data:`CONDENSITE_LGBM_PRESET` value.
    u_max_bin : int, optional
        Number of histogram bins for the auxiliary feature ``u`` (the ``x``
        features use ``regression_params["max_bin"]``). Defaults to the
        :data:`CONDENSITE_LGBM_PRESET` value.
    regression_params : dict, optional
        LightGBM parameters; they override the wrapper's own settings. Defaults
        to the :data:`CONDENSITE_LGBM_PRESET` value.
    prediction_batch_size : int, optional
        Number of expanded ``(x, u)`` rows per prediction batch (bounds memory).
    random_seed : int
        Seed for condensité's auxiliary draws and LightGBM.
    """

    def __init__(
        self,
        h: float | None = None,
        M: int | None = None,
        min_leaf_mass: float | None = None,
        n_grid: int | None = None,
        u_max_bin: int | None = None,
        regression_params: dict | None = None,
        prediction_batch_size: int | None = 100_000,
        random_seed: int = 0,
    ):
        preset = CONDENSITE_LGBM_PRESET
        self.h = float(h if h is not None else preset["h"])
        self.M = int(M if M is not None else preset["M"])
        self.min_leaf_mass = float(min_leaf_mass if min_leaf_mass is not None else preset["min_leaf_mass"])
        self.n_grid = int(n_grid if n_grid is not None else preset["n_grid"])
        self.u_max_bin = int(u_max_bin if u_max_bin is not None else preset["u_max_bin"])
        if regression_params is not None:
            self.regression_params = regression_params
        else:
            self.regression_params = dict(preset["regression_params"])
        self.prediction_batch_size = prediction_batch_size
        self.random_seed = int(random_seed)

        self._model = None

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "CondensiteLGBMWrapper":
        try:
            import condensite as cde
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import condensite. Install condensite>=0.0.2 to use this wrapper."
            ) from exc

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")
        self._set_train_range(y)

        # A leaf spanning k observations and a u-interval of width w holds about
        # k*M*w expanded rows, so min_data_in_leaf = min_leaf_mass*M enforces
        # k*w >= min_leaf_mass. As for a histogram bin, the number of observations
        # whose ỹ falls into the leaf's u-interval is about k*w*p̃(ỹ | x).
        # u is binned more finely than x: its bin width 1/u_max_bin must stay
        # well below h for the trees to resolve the kernel.
        params = {
            "objective": "regression",
            "min_data_in_leaf": max(1, round(self.min_leaf_mass * self.M)),
            "seed": self.random_seed,
            "verbosity": -1,
            **self.regression_params,
        }
        x_max_bin = params.pop("max_bin", 255)
        params["max_bin_by_feature"] = [x_max_bin] * X.shape[1] + [self.u_max_bin]

        # condensité draws the auxiliary points from the global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            self._model = cde.Condensite(
                _LGBMPredictor(params),
                h=self.h,
                M=self.M,
                prediction_batch_size=self.prediction_batch_size,
            )
            self._model.fit(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                train_frac=1.0,
                n_grid_ISE=None,
            )
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        X = torch.as_tensor(self._sanitize_X(X), dtype=torch.float32)
        cdes, y_points = self._model.predict(X, n_grid=self.n_grid)
        return grid_density_to_distribution(
            y_points.numpy().astype(np.float64).reshape(-1),
            cdes.numpy().astype(np.float64),
            train_range=self._y_train_range,
        )

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean
