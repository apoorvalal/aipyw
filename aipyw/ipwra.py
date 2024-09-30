from typing import List, Optional
import numpy as np

from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

class IpwRa:
    def __init__(self, estimand: str = "ATE", typ: str = "ipwra"):
        self.estimand = estimand
        self.typ = typ
        self.point_estimate = None
        self.bootstrap_results = None

    def fit(self, df: pd.DataFrame, outcome: str = "Y", treatment: str = "W",
            covariates: Optional[List[str]] = None, alpha: float = 0.05,
            n_iterations: int = 100, n_jobs: int = -1):
        if covariates is None:
            covariates = df.columns.difference([outcome, treatment]).tolist()

        y,w,X = df[outcome].values, df[treatment].values, df[covariates].values

        self.point_estimate = self._ipwra(y, w, X)

        if n_iterations > 0:
            self._bootstrap(df, outcome, treatment, covariates, n_iterations, n_jobs)

        return self.summary(alpha)

    def summary(self, alpha: float = 0.05):
        result = {"est": round(self.point_estimate, 4)}

        if self.bootstrap_results is not None:
            lb, ub = np.percentile(
                self.bootstrap_results,
                [alpha / 2 * 100, (1 - alpha / 2) * 100]
            )
            result[f"{(1-alpha)*100}% CI (Bootstrap)"] = np.round([lb, ub], 4)

        return result
    ######################################################################

    def _ipwra(self, y: np.ndarray, w: np.ndarray, X: np.ndarray):
        # underlying function
        sca = StandardScaler()
        Xtilde = sca.fit_transform(X)
        if self.typ in ["ipwra", "ipw"]:
            lr = LogisticRegression(C=1e12)
            lr.fit(X, w)
            p = lr.predict_proba(X)
            if self.estimand == "ATE":
                wt = w / p[:, 1] + (1 - w) / p[:, 0]
            elif self.estimand == "ATT":
                rho = np.mean(w)
                wt = p[:, 1] / rho + (1 - w) / p[:, 0]

        if self.typ in ["ipwra", "ra"]:
            XX = np.c_[
                np.ones(len(w)), w, Xtilde, w[:, np.newaxis] * Xtilde
            ]  # design matrix has [1, W, X, W * X] where X is centered
        else:
            XX = np.c_[np.ones(len(w)), w]

        if self.typ == "ra":
            wt = None

        return LinearRegression().fit(XX, y, sample_weight=wt).coef_[1]

    def _bootstrap(self, df: pd.DataFrame, outcome: str, treatment: str,
                   covariates: List[str], n_iterations: int, n_jobs: int):
        # bootstrap confidence interval
        def onerep():
            bsamp = df.sample(n=len(df), replace=True)
            return self._ipwra(bsamp[outcome].values, bsamp[treatment].values, bsamp[covariates].values)

        self.bootstrap_results = np.array(Parallel(n_jobs=n_jobs)(
            delayed(onerep)() for _ in range(n_iterations)
        ))

