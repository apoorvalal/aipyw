import numpy as np
from typing import Tuple
from scipy import optimize


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
