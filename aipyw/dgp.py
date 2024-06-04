import numpy as np
import scipy


def dgp_binary(
    n: int = int(1e5),
    k: int = 10,
    func_e=lambda X: 1 + X[:, 4] + 1 * (X[:, 1] > 0) * X[:, 2],
    func_t=lambda X: X[:, 0] + np.sin(X[:, 1]) + X[:, 2] * X[:, 3],
    func_y=lambda X: X[:, 0] + X[:, 1] + np.sin(X[:, 2]) + np.exp(X[:, 3] * X[:, 4]),
):
    """Data Generating Process for binary treatment
    Generates cross-sectional data for a binary treatment setting with arbitrarily non-linear propensity, outcome, and treatment-effect functions.

    Args:
        n (int, optional): sample size. Defaults to int(1e5).
        k (int, optional): number of covariates. Defaults to 10.
        func_e (lambda, optional): treatment effect effect function. Defaults to 1 + X[:, 4] + 1 * (X[:, 1] > 0) * X[:, 2], which produces heterogeneous effects. Can also be a constant.
        func_t (lambda, optional): lambda for propensity score. Defaults to X[:, 0]+np.sin(X[:, 1])+X[:, 2]*X[:, 3] - which can only be approximated by flexible regression methods. If constant (e.g. 0.5), generates a randomized trial.
        func_y (lambda, optional): outcome function. Defaults to X[:, 0]+X[:, 1]+np.sin(X[:, 2])+np.exp(X[:, 3] * X[:, 4]).

    Returns:
        tuple: np.array of outcomes, np.array of treatments, np.array of covariates
    """
    # covariates
    X = np.random.multivariate_normal(np.zeros(k), np.eye(k), size=n)
    D = np.random.binomial(1, scipy.special.expit(func_t(X)))
    Y = func_y(X) + D * (func_e(X)) + np.random.normal(size=(n,))
    return Y.reshape(-1, 1), D.reshape(-1, 1), X


def dgp_discrete(
    n: int = int(1e5),
    k: int = 10,
    p: int = 3,
    treat_effects: np.array = np.array([0.1, 0.5, 0.25]),
):
    """Generate data for discrete treatment setting

    Args:
        n (int, optional): sample size. Defaults to int(1e5).
        k (int, optional): covariate dimensions. Defaults to 10.
        p (int, optional): number of treatments. Defaults to 3.
        treat_effects (np.array, optional): potential outcome means; needs to be np.array of length p. Defaults to np.array([0.1, 0.5, 0.25]).

    Returns:
        tuple: np.array of outcomes, np.array of treatments, np.array of covariates
    """
    # generate covariates
    X = np.random.multivariate_normal(np.zeros(k), np.eye(k), size=n)

    # propensity score
    def multinomial_logit(X, coefficients):
        "multinomial logit of X @ coefficients.T"
        xb = X @ coefficients.T
        elogit = np.exp(xb)
        return elogit / elogit.sum(axis=1, keepdims=True)

    pscore_coefs = np.random.randn(p, X.shape[1])
    β = np.random.randn(1, k).reshape(-1, 1)
    psmat = multinomial_logit(X, pscore_coefs)
    D = np.array([np.random.choice(p, p=prob) for prob in psmat])
    # Calculate outcome
    Y = (X @ β).ravel() + treat_effects[D] + np.random.normal(0, 0.1, size=(n))
    return Y.reshape(-1, 1), D.reshape(-1, 1), X


def dgp_continuous(
    n: int = 10_000,
    k: int = 5,
    func_e=lambda D, X: 0.5 * D**2,
    func_y=lambda X: X[:, 1],
    func_t=lambda X: np.abs(X[:, 3]),
):
    X = np.random.normal(size=(n, k), scale=5)
    D = np.random.normal(loc=func_t(X))
    Y = (
        func_e(D, X)
        + func_y(X)
        + np.random.normal(
            size=n,
        )
        * 20
    )
    return Y.reshape(-1, 1), D.reshape(-1, 1), X


def generate_toeplitz_matrix(k: int, max_value: float = 0.5, min_value: float = 0.0):
    """generate covariance matrices to feed into covariate generation

    Args:
        k (int): number of covariates
        max_value (float, optional): Max covariance. Defaults to 0.5.
        min_value (float, optional): Min covariance. Defaults to 0.00.

    Returns:
        np.array: Toeplitz matrix
    """
    # Generate the first row of the Toeplitz matrix
    first_row = np.linspace(max_value, min_value, k)
    toeplitz_matrix = np.zeros((k, k))
    for i in range(k):
        toeplitz_matrix[i, :] = np.roll(first_row, i)
    toeplitz_matrix += np.eye(k) * 0.5
    return np.round(toeplitz_matrix, 3)
