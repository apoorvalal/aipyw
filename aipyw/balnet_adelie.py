"""Covariate-balancing propensity scores with Adelie.

This module is a small proof-of-concept Python port of the core `balnet`
idea: replace logistic likelihood by a covariate-balancing calibration loss and
solve the penalized path with Adelie's group elastic-net solver.

The implementation intentionally stays close to the balnet formulation for one
arm. `fit_balnet_ate` calls the one-arm fitter twice, once for treated and once
for controls, to produce ATE weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import adelie as ad
import numpy as np


Target = Literal["ATE", "ATT", "treated", "control"]


@dataclass
class Standardization:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asfortranarray((np.asarray(x, dtype=np.float64) - self.center) / self.scale)


@dataclass
class OneArmBalnetFit:
    """One balancing-loss regularized logistic path.

    Parameters are stored on the original covariate scale.  `arm="treated"`
    means the fitted propensity is used for treated weights `W/e(X)`.  `arm="control"`
    means the fitted propensity is used as `e(X)` but control weights use
    `(1-W)/(1-e(X))` or ATT odds weights depending on the caller.
    """

    arm: Literal["treated", "control"]
    lambdas: np.ndarray
    intercepts: np.ndarray
    betas: np.ndarray
    standardization: Standardization
    adelie_state: object

    def linear_predictor(self, x: np.ndarray, lambda_index: int = -1) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.intercepts[lambda_index] + x @ self.betas[lambda_index]

    def predict_proba(self, x: np.ndarray, lambda_index: int = -1) -> np.ndarray:
        eta = np.clip(self.linear_predictor(x, lambda_index=lambda_index), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-eta))


@dataclass
class BalnetATEFit:
    treated: OneArmBalnetFit | None
    control: OneArmBalnetFit | None
    target: Target
    sample_weights: np.ndarray

    @property
    def lambdas(self) -> dict[str, np.ndarray]:
        out = {}
        if self.treated is not None:
            out["treated"] = self.treated.lambdas
        if self.control is not None:
            out["control"] = self.control.lambdas
        return out


def _weighted_mean_sd(x: np.ndarray, weights: np.ndarray) -> Standardization:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    center = weights @ x
    xc = x - center
    scale = np.sqrt(weights @ (xc * xc))
    scale = np.where(scale <= 0, 1.0, scale)
    return Standardization(center=center, scale=scale)


class CBPSGlm(ad.glm.glm_base, ad.glm.GlmBase64):
    """Adelie GLM wrapper for the balnet calibration loss.

    The one-arm loss is

        sum_i w_i [ y_i exp(-eta_i) + (1-y_i) eta_i ],

    with gradient

        w_i [ -y_i exp(-eta_i) + (1-y_i) ]

    for the mathematical objective.  Adelie's GLM interface expects the
    negative gradient in its `gradient` method, matching balnet's C++ extension:

        w_i [ y_i exp(-eta_i) - (1-y_i) ].
    """

    def __init__(self, y: np.ndarray, weights: np.ndarray | None = None, target_scale: float = 1.0):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if weights is None:
            weights = np.full(y.shape[0], 1.0 / y.shape[0], dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()
        self.target_scale = float(target_scale)
        ad.glm.glm_base.__init__(self, y, weights, ad.glm.GlmBase64, np.float64)
        ad.glm.GlmBase64.__init__(self, "cbps", self.y, self.weights)

    def gradient(self, eta: np.ndarray, out: np.ndarray) -> None:
        eta = np.clip(eta, -35.0, 35.0)
        out[:] = self.target_scale * self.weights * (self.y * np.exp(-eta) - (1.0 - self.y))

    def hessian(self, eta: np.ndarray, grad: np.ndarray, out: np.ndarray) -> None:
        # Algebraically equal to target_scale * weights * y * exp(-eta), but this
        # matches balnet's implementation and reuses the gradient already computed.
        out[:] = grad + self.target_scale * self.weights * (1.0 - self.y)
        out[:] = np.maximum(out, 1e-12)

    def loss(self, eta: np.ndarray) -> float:
        eta = np.clip(eta, -35.0, 35.0)
        return float(np.sum(self.target_scale * self.weights * (self.y * np.exp(-eta) + (1.0 - self.y) * eta)))

    def loss_full(self) -> float:
        # Not used for the balancing-loss interpretation.
        return 0.0

    def inv_link(self, eta: np.ndarray, out: np.ndarray) -> None:
        eta = np.clip(eta, -35.0, 35.0)
        out[:] = 1.0 / (1.0 + np.exp(-eta))


def _unstandardize_path(
    betas_std: np.ndarray,
    intercepts_std: np.ndarray,
    standardization: Standardization,
) -> tuple[np.ndarray, np.ndarray]:
    betas = betas_std / standardization.scale[None, :]
    intercepts = intercepts_std - betas_std @ (standardization.center / standardization.scale)
    return intercepts, betas


def fit_balnet_arm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    arm: Literal["treated", "control"] = "treated",
    sample_weights: np.ndarray | None = None,
    standardize_with: np.ndarray | None = None,
    lambdas: np.ndarray | None = None,
    n_lambdas: int = 60,
    min_ratio: float = 1e-2,
    alpha: float = 1.0,
    penalty: np.ndarray | None = None,
    progress_bar: bool = False,
    max_iters: int = 100_000,
    tol: float = 1e-7,
    n_threads: int = 1,
) -> OneArmBalnetFit:
    """Fit one arm of the balnet calibration-loss path.

    `y` should be the arm indicator used by the one-arm loss. For an ATE fit,
    call with `y=W` for the treated arm and `y=1-W` for the control arm.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    if y.shape != (x.shape[0],):
        raise ValueError("y must have length nrow(x)")
    if sample_weights is None:
        sample_weights = np.ones(x.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if standardize_with is None:
        standardize_with = sample_weights
    stan = _weighted_mean_sd(x, standardize_with)
    xs = stan.transform(x)

    glm = CBPSGlm(y, weights=sample_weights)
    state = ad.grpnet(
        xs,
        glm,
        alpha=alpha,
        penalty=penalty,
        lmda_path=lambdas,
        lmda_path_size=n_lambdas,
        min_ratio=min_ratio,
        max_iters=max_iters,
        tol=tol,
        adev_tol=0.9,
        ddev_tol=0,
        early_exit=False,
        intercept=True,
        progress_bar=progress_bar,
        n_threads=n_threads,
    )
    betas_std = state.betas.toarray()
    intercepts_std = np.asarray(state.intercepts, dtype=np.float64)
    intercepts, betas = _unstandardize_path(betas_std, intercepts_std, stan)
    return OneArmBalnetFit(
        arm=arm,
        lambdas=np.asarray(state.lmdas, dtype=np.float64),
        intercepts=intercepts,
        betas=betas,
        standardization=stan,
        adelie_state=state,
    )


def fit_balnet_ate(
    x: np.ndarray,
    w: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    target: Target = "ATE",
    **kwargs,
) -> BalnetATEFit:
    """Fit balnet-style balancing propensity models for ATE/ATT targets."""

    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if sample_weights is None:
        sample_weights = np.ones(x.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    target = target.upper()  # type: ignore[assignment]

    standardize_with = sample_weights if target != "ATT" else sample_weights * w
    treated = control = None
    if target in {"ATE", "TREATED"}:
        treated = fit_balnet_arm(
            x,
            w,
            arm="treated",
            sample_weights=sample_weights,
            standardize_with=standardize_with,
            **kwargs,
        )
    if target in {"ATE", "ATT", "CONTROL"}:
        control = fit_balnet_arm(
            x,
            1.0 - w,
            arm="control",
            sample_weights=sample_weights,
            standardize_with=standardize_with,
            **kwargs,
        )
    return BalnetATEFit(treated=treated, control=control, target=target, sample_weights=sample_weights)


def balancing_weights(fit: BalnetATEFit, x: np.ndarray, w: np.ndarray, lambda_index: int = -1) -> dict[str, np.ndarray]:
    """Return balnet-style IPW weights at one path index."""

    w = np.asarray(w, dtype=np.float64)
    sw = fit.sample_weights
    out: dict[str, np.ndarray] = {}
    if fit.treated is not None:
        e = np.clip(fit.treated.predict_proba(x, lambda_index=lambda_index), 1e-8, 1.0 - 1e-8)
        wt = np.zeros_like(w, dtype=np.float64)
        wt[w == 1] = sw[w == 1] / e[w == 1]
        out["treated"] = wt
    if fit.control is not None:
        e_control_model = np.clip(fit.control.predict_proba(x, lambda_index=lambda_index), 1e-8, 1.0 - 1e-8)
        # The control model is fit with y=1-W. balnet returns predict(control)=1-p_control_model.
        e = 1.0 - e_control_model
        wc = np.zeros_like(w, dtype=np.float64)
        if fit.target == "ATT":
            wc[w == 0] = sw[w == 0] * e[w == 0] / (1.0 - e[w == 0])
        else:
            wc[w == 0] = sw[w == 0] / (1.0 - e[w == 0])
        out["control"] = wc
    return out


def standardized_mean_differences(x: np.ndarray, weights: np.ndarray, target_weights: np.ndarray) -> np.ndarray:
    """Weighted mean differences standardized by target weighted SD."""

    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    weights = weights / weights.sum()
    target_weights = target_weights / target_weights.sum()
    mu_w = weights @ x
    mu_t = target_weights @ x
    xc = x - mu_t
    sd_t = np.sqrt(target_weights @ (xc * xc))
    sd_t = np.where(sd_t <= 0, 1.0, sd_t)
    return (mu_w - mu_t) / sd_t


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    s1 = weights.sum()
    s2 = np.sum(weights * weights)
    return float(s1 * s1 / s2) if s2 > 0 else 0.0
