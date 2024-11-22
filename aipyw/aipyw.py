import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler

from .weights import balancing_weights
from .calibrate import isoreg_with_xgboost


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
        bal_obj=None,
        calibrate=False,
    ):
        """
        Initialize the AIpyw class.

        Parameters:
        - propensity_model: The model used to estimate the propensity scores. Default is None, which uses LogisticRegression.
        - outcome_model: The model used to estimate the outcome regression. Default is None, which uses LinearRegression.
        - balance_model: The model used to estimate the balance regression. Default is None, which uses RidgeCV.
        - riesz_method: The method used for Riesz balancing. Can be 'ipw', 'linear', 'kernel', 'automatic' or "balancing". Default is 'ipw'.
        - n_splits: The number of splits for cross-validation. Default is 2.
        - bal_order: The order of balance polynomial under linear balancing weights. Default is None, which uses 2.
        - bal_obj: The objective function for balancing weights. Default is None, which uses 'entropy'.
        - calibrate : Whether to calibrate the nuisance models. Default is False.
        """

        self.n_splits = n_splits
        # nuisance models
        self.propensity_model = propensity_model or LogisticRegression()
        self.outcome_model = outcome_model or LinearRegression()
        self.balance_model = balance_model or RidgeCV()
        self.riesz_method = riesz_method or "ipw-hajek"
        self._bal_order = bal_order or 2
        self._bal_obj = bal_obj or "quadratic"
        self._calibrate = calibrate

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
        self.treatment_levels = np.unique(W)
        self.K = len(self.treatment_levels)
        self._n_rff = n_rff or self.n // 5  # default to n/5 fourier features
        # Cross-fit outcome models for each treatment
        self.mu_hat = self._cross_fit_outcome(X, W, Y)
        # Estimate Riesz representer
        self.a_x = self._estimate_riesz_representer(X, W)
        # Calculate AIPW estimates
        self.calculate_aipw_estimates()
        return self

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

    def summary(self, prec=5):
        effects = {}
        # all pairwise contrasts: nests binary ATE case
        for i in range(self.K):
            for j in range(i + 1, self.K):
                psi_ij = self.influence_functions[:, j] - self.influence_functions[:, i]
                effect, se = np.mean(psi_ij), np.std(psi_ij) / np.sqrt(self.n)
                effects[f"{j} vs {i}"] = {
                    "effect": np.round(effect, prec),
                    "se": np.round(se, prec),
                }
        return effects

    ######################################################################
    # riesz representer estimation block
    def _estimate_riesz_representer(self, X, W):
        """Internal method to estimate the Riesz representer."""
        a_x = np.zeros((len(W), self.K))

        if self.riesz_method in [
            "ipw",
            "ipw-hajek",
        ]:  # farm out to cross-fit propensity
            self.propensity_scores = self._cross_fit_propensity(X, W)
            for w in range(self.K):
                a_x[:, w] = (W == w) / self.propensity_scores[:, w]
                if self.riesz_method == "ipw-hajek":
                    a_x[:, w] = a_x[:, w] / (self.propensity_scores[:, w].sum())
        elif self.riesz_method == "balancing":  # farm out to balancing
            for w in range(self.K):
                a_x[:, w] = self._balancing(X, w)
        elif self.riesz_method in ["linear", "kernel"]:  # fit linear or kernel ridge
            if self.riesz_method == "kernel":

                rks = RBFSampler(gamma=1, n_components=self._n_rff, random_state=42)
                X = rks.fit_transform(X)
            else:  # linear

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
        elif self.riesz_method == "automatic":  # ADML method
            for w in range(self.K):
                mask = W == w
                X_w = X[mask]
                n_w = X_w.shape[0]
                overall_mean = np.mean(X, axis=0)

                Phi_w = np.c_[np.ones(n_w), X_w]
                Phi_q = np.r_[1, overall_mean]

                A = (1 / n_w) * Phi_w.T @ Phi_w + 1e-4 * np.eye(X.shape[1] + 1)
                b = (1 / len(W)) * Phi_q

                theta = np.linalg.solve(A, b)
                a_x[mask, w] = Phi_w @ theta * (len(W) / n_w)
        else:
            raise ValueError(f"Unknown Riesz method: {self.riesz_method}")
        return a_x

    def _cross_fit_propensity(self, X, W):
        """Internal method to cross-fit propensity scores."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        propensity_scores = np.zeros((self.n, self.K))

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            W_train = W[train_index]

            model = clone(self.propensity_model)
            model.fit(X_train, W_train)
            propensity_scores[test_index] = model.predict_proba(X_test)

        if (
            self._calibrate
        ):  # apply isotonic calibration over whole dataset for each treatment level
            for idx, w in enumerate(self.treatment_levels):
                # Create indicator for current treatment level
                W_ind = (W == w).astype(int)

                # Select the corresponding propensity score column
                pi = propensity_scores[:, idx]

                # Calibrate propensity scores using isotonic regression with XGBoost
                calibrator_pi = isoreg_with_xgboost(pi, W_ind)
                pi_star = calibrator_pi(pi)

                # Ensure pi_star values are bounded below to prevent division issues
                c1 = np.min(pi_star[W_ind == 1])
                pi_star = np.maximum(pi_star, c1)

                # Store the calibrated propensity scores in the matrix
                propensity_scores[:, idx] = pi_star

        return propensity_scores

    def _balancing(self, X, w):
        """Internal method to balance covariates using balancing weights."""
        scaler = MinMaxScaler()
        Xw = scaler.fit_transform(X[self.W == w])
        X_all = scaler.transform(X)
        # store deviation from the target covariates
        Z = np.c_[
            np.ones(Xw.shape[0]),
            Xw - np.average(X_all, axis=0),
        ]
        weight_link, beta, status = balancing_weights(
            Z, objective=self._bal_obj, min_weight=0.0, max_weight=1.0, l2_norm=0
        )
        Xmat = np.c_[np.ones(X_all.shape[0]), X_all]
        return weight_link(np.dot(Xmat, beta))

    def _linear_balancing(self, X_train, W_train, X_test, w):
        model = clone(self.balance_model)
        y = (W_train == w).astype(float)
        model.fit(X_train, y)
        return model.predict(X_test)

    ############################################################
    # riesz representer estimation block
    def _cross_fit_outcome(self, X, W, Y):
        """Internal method to cross-fit outcome models."""
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

        if (
            self._calibrate
        ):  # apply isotonic calibration over whole dataset for each treatment level
            for idx, w in enumerate(self.treatment_levels):
                # Create indicator for current treatment level
                W_ind = (W == w).astype(int)
                mu = mu_hat[:, idx]

                # Calibrate outcome model using isotonic regression with XGBoost
                calibrator_mu = isoreg_with_xgboost(mu[W_ind], Y[W_ind])
                mu_star = calibrator_mu(mu)

                # Store the calibrated propensity scores in the matrix
                mu_hat[:, idx] = mu_star

        return mu_hat


######################################################################
