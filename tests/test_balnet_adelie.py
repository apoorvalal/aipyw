import numpy as np
from scipy import optimize

from aipyw.balnet_adelie import (
    CBPSGlm,
    balancing_weights,
    effective_sample_size,
    fit_balnet_arm,
    fit_balnet_ate,
    standardized_mean_differences,
)


def _simulate(seed=0, n=600, p=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    eta = -0.15 + x[:, 0] - 0.6 * x[:, 1] + 0.35 * x[:, 2]
    e = 1.0 / (1.0 + np.exp(-eta))
    w = rng.binomial(1, e).astype(float)
    return x, w


def test_cbps_glm_derivatives_have_expected_signs():
    y = np.array([1.0, 0.0, 1.0])
    glm = CBPSGlm(y)
    eta = np.zeros_like(y)
    grad = np.empty_like(y)
    hess = np.empty_like(y)
    glm.gradient(eta, grad)
    glm.hessian(eta, grad, hess)
    assert np.allclose(grad, np.array([1 / 3, -1 / 3, 1 / 3]))
    assert np.all(hess > 0)


def test_one_arm_matches_scipy_lasso_objective_at_fixed_lambda():
    x, w = _simulate(seed=1, n=160, p=4)
    lam = 0.04
    fit = fit_balnet_arm(x, w, lambdas=np.array([lam]), n_lambdas=1, progress_bar=False)
    beta0 = fit.intercepts[0]
    beta = fit.betas[0]

    stan = fit.standardization
    xs = stan.transform(x)
    weights = np.full(x.shape[0], 1.0 / x.shape[0])

    def obj(theta):
        a = theta[0]
        b = theta[1:]
        eta = np.clip(a + xs @ b, -35, 35)
        return np.sum(weights * (w * np.exp(-eta) + (1 - w) * eta)) + lam * np.sum(np.abs(b))

    theta0 = np.r_[fit.adelie_state.intercepts[0], fit.adelie_state.betas.toarray()[0]]
    res = optimize.minimize(obj, theta0, method="Powell", options={"xtol": 1e-8, "ftol": 1e-8, "maxiter": 3000})
    # Compare on the original prediction scale; Powell may choose a slightly
    # different point at inactive-coordinate kinks but the fitted eta should agree.
    eta_adelie = beta0 + x @ beta
    b_s = res.x[1:] / stan.scale
    a_s = res.x[0] - res.x[1:] @ (stan.center / stan.scale)
    eta_scipy = a_s + x @ b_s
    assert np.sqrt(np.mean((eta_adelie - eta_scipy) ** 2)) < 5e-4


def test_ate_weights_reduce_balance_against_overall_sample():
    x, w = _simulate(seed=2, n=800, p=8)
    fit = fit_balnet_ate(x, w, n_lambdas=18, min_ratio=1e-2, progress_bar=False)
    bw = balancing_weights(fit, x, w, lambda_index=-1)
    target = np.ones_like(w)

    treated_before = np.max(np.abs(standardized_mean_differences(x, w, target)))
    control_before = np.max(np.abs(standardized_mean_differences(x, 1 - w, target)))
    treated_after = np.max(np.abs(standardized_mean_differences(x, bw["treated"], target)))
    control_after = np.max(np.abs(standardized_mean_differences(x, bw["control"], target)))

    assert treated_after < treated_before / 5
    assert control_after < control_before / 5
    assert effective_sample_size(bw["treated"]) > 0
    assert effective_sample_size(bw["control"]) > 0
