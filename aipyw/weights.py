import numpy as np
from typing import Literal, Tuple
from scipy import optimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


def balancing_weights(
    z: np.ndarray,
    objective: str = "entropy",
    min_weight: float = 0.0,
    max_weight: float = 10.0,
    l2_norm: float = 0,
) -> Tuple[np.ndarray, bool]:
    """Calibrates covariates toward target.

    solves a constrained convex optimization problem that minimizes the
    variation of weights for units while achieving direct covariate
    balance. The weighted mean of covariates would match the simple mean
    of target covariates up to a prespecified L2 norm.
    There are two choices of the optimization objective: entropy of the weights
    (entropy balancing, or EB) and effective sample size implied by the weights
    (quadratic balancing, or QB). EB can be viewed as minimizing the
    Kullback-Leibler divergence between the optimal weights and equal weights;
    while QB effectively minimizes the Euclidean distance between the optimal
    weights and equal weights. The two objectives correspond to different link
    functions for the weights (or the odds of propensity scores) - `exp(x)` for EB
    and `max(x, 0)` for QB. Therefore, EB weights are strictly positive; while QB
    weights can be zero and induce sparsity.

    Args:
      z : Matrix of starting weights. X0 - X1bar
      objective: The objective of the convex optimization problem. Supported
        values are "entropy" and "quadratic".
      min_weight: The lower bound on weights. Must be between 0.0 and the uniform
        weight (1 / number of rows in `covariates`).
      max_weight: The upper bound on weights. Must be between the uniform weight
        (1 / number of rows in `covariates`) and 1.0.
      l2_norm: The L2 norm of the covaraite balance constraint, i.e., the
        Euclidean distance between the weighted mean of covariates and the simple
        mean of target covaraites after balancing.
    """
    n, k = z.shape
    k -= 1
    if objective == "entropy":
        weight_link = lambda x: np.exp(np.minimum(x, np.log(1e8)))
        beta_init = np.zeros(k + 1)
    elif objective == "quadratic":
        weight_link = lambda x: np.clip(x, min_weight, max_weight)
        beta_init = np.linalg.pinv(z.T @ z) @ np.concatenate((np.ones(1), np.zeros(k)))
    else:
        raise ValueError(f"Unknown objective: {objective}")

    def estimating_equation(beta):
        weights = weight_link(np.dot(z, beta))
        norm = np.linalg.norm(beta[1:])
        if norm == 0.0:
            slack = np.zeros(len(beta[1:]))
        else:
            slack = l2_norm * beta[1:] / norm
        return np.dot(z.T, weights) + np.concatenate((-np.ones(1), slack))

    beta, info_dict, status, msg = optimize.fsolve(
        estimating_equation, x0=beta_init, full_output=True
    )
    weights = weight_link(np.dot(z, beta))
    # ebal: recompute weight if constraints violated
    if objective == "entropy" and (
        (np.max(weights) > max_weight) or (np.min(weights) < min_weight)
    ):
        if min_weight == 0.0:
            weight_link = lambda x: np.exp(np.minimum(x, np.log(max_weight)))
        else:
            weight_link = lambda x: np.exp(
                np.clip(x, np.log(min_weight), np.log(max_weight))
            )
        beta, info_dict, status, msg = optimize.fsolve(
            estimating_equation, x0=beta, full_output=True
        )
    return weight_link, beta, status


class CovariateBalancer(BaseEstimator, TransformerMixin):
    """Estimate covariate-balancing weights from a source sample to a target sample.

    This is a small sklearn-compatible wrapper around the calibration routines in
    this module. It is intentionally causal-agnostic: it only knows about a
    source covariate distribution and a target covariate distribution. Causal
    estimators can use it by fitting one balancer per treatment arm and then
    masking the resulting Riesz representer outside that arm.
    """

    def __init__(
        self,
        method: Literal["quadratic", "entropy", "balnet"] = "quadratic",
        *,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        l2_norm: float = 0.0,
        scale: bool = True,
        lambda_index: int = -1,
        n_lambdas: int = 60,
        min_ratio: float = 1e-2,
        alpha: float = 1.0,
        progress_bar: bool = False,
        max_iters: int = 100_000,
        tol: float = 1e-7,
        n_threads: int = 1,
    ):
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.l2_norm = l2_norm
        self.scale = scale
        self.lambda_index = lambda_index
        self.n_lambdas = n_lambdas
        self.min_ratio = min_ratio
        self.alpha = alpha
        self.progress_bar = progress_bar
        self.max_iters = max_iters
        self.tol = tol
        self.n_threads = n_threads

    def fit(self, X, y=None, *, target_X=None, sample_weight=None, target_weight=None):
        """Fit source-to-target balancing weights.

        ``y`` is kept for sklearn compatibility. If ``target_X`` is omitted and
        ``y`` is two-dimensional, ``y`` is interpreted as the target covariate
        matrix so ``fit(source_X, target_X)`` works naturally.
        """

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if target_X is None and y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 2:
                target_X = y_arr
        if target_X is None:
            target_X = X
        target_X = np.asarray(target_X, dtype=float)
        if target_X.ndim != 2 or target_X.shape[1] != X.shape[1]:
            raise ValueError("target_X must be two-dimensional with the same number of columns as X")

        self.n_features_in_ = X.shape[1]
        self.source_shape_ = X.shape
        self.target_shape_ = target_X.shape

        if self.scale:
            self.scaler_ = MinMaxScaler().fit(X)
            Xs = self.scaler_.transform(X)
            Xt = self.scaler_.transform(target_X)
        else:
            self.scaler_ = None
            Xs, Xt = X, target_X

        if target_weight is None:
            target_weight = np.ones(Xt.shape[0], dtype=float)
        target_weight = np.asarray(target_weight, dtype=float)
        target_weight = target_weight / target_weight.sum()
        self.target_ = target_weight @ Xt

        method = self.method.lower()
        if method in {"quadratic", "entropy"}:
            Z = np.c_[np.ones(Xs.shape[0]), Xs - self.target_]
            self.weight_link_, self.beta_, self.status_ = balancing_weights(
                Z,
                objective=method,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                l2_norm=self.l2_norm,
            )
        elif method == "balnet":
            from .balnet_adelie import fit_balnet_arm

            if sample_weight is None:
                source_weight = np.ones(Xs.shape[0], dtype=float)
            else:
                source_weight = np.asarray(sample_weight, dtype=float)
            # Normalize source and target masses equally so the intercept score
            # encodes source-weight mass = target-weight mass.
            source_weight = 0.5 * source_weight / source_weight.sum()
            target_weight_balnet = 0.5 * target_weight / target_weight.sum()
            x_stack = np.vstack([Xs, Xt])
            y_stack = np.r_[np.ones(Xs.shape[0]), np.zeros(Xt.shape[0])]
            w_stack = np.r_[source_weight, target_weight_balnet]
            self.balnet_fit_ = fit_balnet_arm(
                x_stack,
                y_stack,
                sample_weights=w_stack,
                standardize_with=w_stack,
                n_lambdas=self.n_lambdas,
                min_ratio=self.min_ratio,
                alpha=self.alpha,
                progress_bar=self.progress_bar,
                max_iters=self.max_iters,
                tol=self.tol,
                n_threads=self.n_threads,
            )
            source_eta = self.balnet_fit_.linear_predictor(Xs, lambda_index=self.lambda_index)
            source_raw = np.exp(np.clip(-source_eta, -35.0, 35.0))
            self.balnet_normalizer_ = source_raw.sum()
            self.status_ = 1
        else:
            raise ValueError(f"Unknown balancing method: {self.method}")
        return self

    def _transform_covariates(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of columns")
        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X

    def transform(self, X):
        """Return balancing weights for rows in ``X``.

        This method returns a one-dimensional weight vector rather than a
        transformed design matrix. That is the most useful sklearn-compatible
        convention for downstream weighting pipelines.
        """

        Xs = self._transform_covariates(X)
        method = self.method.lower()
        if method in {"quadratic", "entropy"}:
            Z = np.c_[np.ones(Xs.shape[0]), Xs - self.target_]
            return self.weight_link_(Z @ self.beta_)
        if method == "balnet":
            eta = self.balnet_fit_.linear_predictor(Xs, lambda_index=self.lambda_index)
            raw = np.exp(np.clip(-eta, -35.0, 35.0))
            return raw / self.balnet_normalizer_
        raise ValueError(f"Unknown balancing method: {self.method}")

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

    def weights(self, X):
        """Alias for ``transform``."""

        return self.transform(X)

    def balance_summary(self, X=None, *, target_X=None, weights=None, target_weight=None):
        """Return mean-difference diagnostics on the scaled balance basis."""

        if X is None:
            raise ValueError("Pass source X explicitly")
        Xs = self._transform_covariates(X)
        if weights is None:
            weights = self.transform(X)
        weights = np.asarray(weights, dtype=float)
        raw_sum = weights.sum()
        weights = weights / raw_sum
        if target_X is None:
            target = self.target_
        else:
            Xt = self._transform_covariates(target_X)
            if target_weight is None:
                target_weight = np.ones(Xt.shape[0], dtype=float)
            target_weight = np.asarray(target_weight, dtype=float)
            target_weight = target_weight / target_weight.sum()
            target = target_weight @ Xt
        diff = weights @ Xs - target
        return {
            "max_abs_mean_difference": float(np.max(np.abs(diff))),
            "mean_abs_mean_difference": float(np.mean(np.abs(diff))),
            "sum_weights": float(raw_sum),
        }
