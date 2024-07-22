import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.model_selection import KFold


class AIPyW:
    r"""Augmented Propensity Score Weighting for many discrete treatments.

    Class to fit the Augmented IPW estimator using arbitrary scikit learners.
    Extends the standard binary treatement estimator to K discrete treatments.
    For details on the influence function, see Cattaneo (2010) JoE.
    """

    def __init__(
        self,
        propensity_model=None,
        outcome_model=None,
        balance_model=None,
        riesz_method="ipw",
        n_splits=2,
        bal_order=None,
    ):
        """
        Initialize the AIpyw class.

        Parameters:
        - propensity_model: The model used to estimate the propensity scores. Default is None, which uses LogisticRegression.
        - outcome_model: The model used to estimate the outcome regression. Default is None, which uses LinearRegression.
        - balance_model: The model used to estimate the balance regression. Default is None, which uses RidgeCV.
        - riesz_method: The method used for Riesz balancing. Can be 'ipw', 'linear', or 'kernel'. Default is 'ipw'.
        - n_splits: The number of splits for cross-validation. Default is 2.
        - bal_order: The order of balance polynomial under linear balancing weights. Default is None, which uses 2.
        """

        self.n_splits = n_splits
        # nuisance models
        self.propensity_model = propensity_model or LogisticRegression()
        self.outcome_model = outcome_model or LinearRegression()
        self.balance_model = balance_model or RidgeCV()
        self.riesz_method = riesz_method or "ipw"
        self._bal_order = bal_order or 2

    def fit(self, X, W, Y, n_rff=None):
        """
        Fits the AIPYW model to the given data.

        Parameters:
        X (array-like: N X K): The feature matrix.
        W (array-like: N-vector): The treatment variable.
        Y (array-like: N-vector): The outcome variable.

        Returns:
        self: The fitted AIPYW model.
        """
        self.X, self.W, self.Y = X, W, Y
        self.n, self.p = X.shape
        self.K = len(np.unique(W))
        self._n_rff = n_rff or self.n // 5  # default to n/5 fourier features
        # Cross-fit outcome models for each treatment
        self.mu_hat = self._cross_fit_outcome(X, W, Y)
        # Estimate Riesz representer
        self.a_x = self._estimate_riesz_representer(X, W)
        # Calculate AIPW estimates
        self.calculate_aipw_estimates()
        return self

    def _cross_fit_propensity(self, X, W):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        propensity_scores = np.zeros((self.n, self.K))

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            W_train = W[train_index]

            model = clone(self.propensity_model)
            model.fit(X_train, W_train)
            propensity_scores[test_index] = model.predict_proba(X_test)

        return propensity_scores

    def _estimate_riesz_representer(self, X, W):
        a_x = np.zeros((len(W), self.K))

        if self.riesz_method == "ipw":
            # fit pscore
            self.propensity_scores = self._cross_fit_propensity(X, W)
            # invert it
            for w in range(self.K):
                a_x[:, w] = (W == w) / self.propensity_scores[:, w]
        elif self.riesz_method in ["linear", "kernel"]:
            if self.riesz_method == "kernel":
                # nystrom approximation
                from sklearn.kernel_approximation import RBFSampler

                rks = RBFSampler(gamma=1, n_components=self._n_rff, random_state=42)
                X = rks.fit_transform(X)
            else:
                from sklearn.preprocessing import PolynomialFeatures

                poly = PolynomialFeatures(degree=self._bal_order, include_bias=False)
                X = poly.fit_transform(X)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                W_train, _ = W[train_index], W[test_index]

                for w in range(self.K):
                    a_x[test_index, w] = self._linear_balancing(
                        X_train, W_train, X_test, w
                    )
        return a_x

    def _linear_balancing(self, X_train, W_train, X_test, w):
        model = clone(self.balance_model)
        y = (W_train == w).astype(float)
        model.fit(X_train, y)
        return model.predict(X_test)

    def _riesz_balancing(self, X_train, W_train, X_test, w):
        # !TODO implement Riesz loss
        pass


    def _cross_fit_outcome(self, X, W, Y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        mu_hat = np.zeros((len(Y), self.K))

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            W_train, Y_train = W[train_index], Y[train_index]

            for w in range(self.K):
                # train outcome model for cell K only on units in cell K
                mask_train = (W_train == w).flatten()  # accommodate for 2D W
                model = clone(self.outcome_model)
                model.fit(X_train[mask_train], Y_train[mask_train])
                # predict on everyone
                mu_hat[test_index, w] = model.predict(X_test)

        return mu_hat

    def calculate_aipw_estimates(self):
        self.aipw_estimates = {}
        self.influence_functions = np.zeros((len(self.Y), self.K))

        for w in range(self.K):
            # Uncentered influence function: mu(x) + a(x) * (y - mu(x))
            self.influence_functions[:, w] = self.mu_hat[:, w] + self.a_x[:, w] * (
                self.Y - self.mu_hat[:, w]
            )

            # AIPW estimate of marginal mean
            marginal_mean = np.mean(self.influence_functions[:, w])
            marginal_se = np.std(self.influence_functions[:, w]) / np.sqrt(len(self.Y))
            self.aipw_estimates[f"{w}"] = {"estimate": marginal_mean, "se": marginal_se}

    def summary(self):
        effects = {}
        # all pairwise contrasts: nests binary ATE case
        for i in range(self.K):
            for j in range(i + 1, self.K):
                psi_ij = self.influence_functions[:, j] - self.influence_functions[:, i]
                effect, se = np.mean(psi_ij), np.std(psi_ij) / np.sqrt(self.n)
                effects[f"{j} vs {i}"] = {"effect": effect, "se": se}
        return effects
