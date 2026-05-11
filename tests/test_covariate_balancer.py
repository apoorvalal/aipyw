import numpy as np
from sklearn.base import clone

from aipyw import CovariateBalancer


def _shifted_samples(seed=0, n_source=250, n_target=400, p=4):
    rng = np.random.default_rng(seed)
    source = rng.normal(loc=0.6, scale=1.0, size=(n_source, p))
    target = rng.normal(loc=0.0, scale=1.0, size=(n_target, p))
    return source, target


def test_covariate_balancer_is_sklearn_cloneable():
    bal = CovariateBalancer(method="entropy", max_weight=5.0)
    cloned = clone(bal)
    assert cloned.get_params()["method"] == "entropy"
    assert cloned.get_params()["max_weight"] == 5.0


def test_direct_covariate_balancer_matches_target_mean():
    source, target = _shifted_samples(seed=1)
    bal = CovariateBalancer(method="quadratic")
    weights = bal.fit_transform(source, target_X=target)
    summary = bal.balance_summary(source, target_X=target, weights=weights)
    assert np.isclose(weights.sum(), 1.0, atol=1e-7)
    assert summary["max_abs_mean_difference"] < 1e-7


def test_entropy_covariate_balancer_accepts_fit_source_target_positional():
    source, target = _shifted_samples(seed=2)
    bal = CovariateBalancer(method="entropy", max_weight=10.0)
    weights = bal.fit(source, target).transform(source)
    summary = bal.balance_summary(source, target_X=target, weights=weights)
    assert np.all(weights > 0)
    assert summary["max_abs_mean_difference"] < 1e-6


def test_balnet_covariate_balancer_reduces_mean_difference():
    source, target = _shifted_samples(seed=3, n_source=500, n_target=500, p=5)
    bal = CovariateBalancer(method="balnet", n_lambdas=20, min_ratio=1e-2, progress_bar=False)
    weights = bal.fit(source, target_X=target).transform(source)
    before = np.max(np.abs(source.mean(axis=0) - target.mean(axis=0)))
    after = np.max(np.abs((weights / weights.sum()) @ source - target.mean(axis=0)))
    assert np.isclose(weights.sum(), 1.0, atol=1e-7)
    assert after < before / 3
