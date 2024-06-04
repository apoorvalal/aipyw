# %%
import numpy as np
import pandas as pd
import sklearn


# %%
class AIPyW:
    r"""Augmented Propensity Score Weighting for many discrete treatments.

    Class to fit the Augmented IPW estimator using arbitrary scikit learners.
    Extends the standard binary treatement estimator to K discrete treatments.
    For details on the influence function, see Cattaneo (2010) JoE.
    """

    def __init__(self, y, w, X, omod, pmod, nf=2, pslb=None):
        """Initialise an aipyw class that holds data and models.

        Args:
                                        y (N X 1 Numpy Array): Response vector
                                        w (N X 1 Numpy Array): Treatment vector (integer valued)
                                        X (N X K Numpy Array): Covariate Matrix
                                        omod (sklearn model object): Model object with .fit() and .predict() methods
                                        pmod (sklearn model object): Model object with .fit() and .predict_proba() methods
                                        nf (int, optional): Number of folds for cross-fitting. Interpreted as no cross-fitting if nf=1 is passed. Defaults to 5.
                                        pslb (Float, optional): Lower-bound for propensity score. Use for trimming extreme pscore values.
        """
        self.y = y
        self.w = w.astype(int)  # need integer for indexing later
        self.X = X
        self.w_levels = np.unique(self.w)
        self.pslb = pslb
        self.psthresh = self.pslb if self.pslb else 0.02

        # missingness
        # !TODO: add missing data indicator and inverse selection weights

        # containers for nuisance functions
        self.K, self.N = len(self.w_levels), len(self.w)
        self.mu = np.empty((self.N, self.K))
        self.pi = np.empty((self.N, self.K))

        # crossfit or not
        if nf > 1:
            self.crossfit = True
            self.nf = nf
        else:
            self.crossfit = False
            self.nf = 1

        # nuisance function machine learners
        self.omod = omod
        self.pmod = pmod

    def fit(self):
        ##################################################################
        # Nuisance functions
        ##################################################################
        if self.crossfit:
            # fit nuisance fns using cross-fitting
            kf = sklearn.model_selection.KFold(
                n_splits=self.nf, shuffle=True, random_state=42
            )
            # iterate over folds
            for testI, trainI in kf.split(self.X):
                # iterate over treatments to fit outcome models
                for w_lev in self.w_levels:
                    muWobs = np.intersect1d(np.where(self.w == w_lev), trainI)
                    # fit outcome model
                    self.omod.fit(self.X[muWobs, :], self.y[muWobs])
                    # predict on held-out fold
                    self.mu[testI, w_lev] = self.omod.predict(self.X[testI, :])
                # propensity model
                self.pmod.fit(self.X[trainI, :], self.w[trainI])
                # predict on held-out fold
                self.pi[testI, :] = self.pmod.predict_proba(self.X[testI, :])
        else:
            for w_lev in self.w_levels:
                muWobs = self.w == w_lev
                # fit outcome model
                self.omod.fit(self.X[muWobs, :], self.y[muWobs])
                # predict on held-out fold
                self.mu[:, w_lev] = self.omod.predict(self.X)
            # propensity model
            self.pmod.fit(self.X, self.w)
            # predict on held-out fold
            self.pi = self.pmod.predict_proba(self.X)

        ##################################################################
        # compute imputed potential outcomes under each treatment
        ##################################################################
        # matrix of treatments [N X K]
        self.wmat = np.repeat(self.w, self.K).reshape(self.N, self.K)
        # matrix of outcomes [N X K]
        self.ymat = np.repeat(self.y, self.K).reshape((self.N, self.K))
        # repeated matrix for each treatment level [N X K]
        self.wdums = np.repeat(np.arange(self.K).reshape(1, self.K), self.N, axis=0)
        ## final computation on N X K matrices
        self.ifvals = (
            1 * (self.wdums == self.wmat) * (self.ymat - self.mu) / self.pi + self.mu
        )
        if np.any(self.pi < self.psthresh):
            print(
                f"Poor overlap - some pscores are < {self.psthresh}; Either call summary() with a trimming threshold as lb \n or change the estimand to ATT."
            )

    def summary(self, lb=None, critval=1.96):
        # !TODO add argument to target ATT, which uses realised outcomes for treated units and differences imputed POs.
        """summarise aipyw model fit. Computes causal contrasts between 0th level (assumed to be control)
                and each other value of w (i.e. K-1 treatment effects if w has K levels).

        Args:
                lb (float, optional): 				Lower bound on propensity score for trimming.
                                                                                        Defaults to None, which corresponds with ATE. Nonzero values no longer target ATE.
                critval (float, optional): 			Normal distribution critical values for confidencei intervals. Defaults to 1.96.
        """
        if lb or self.pslb:  # pscore trimming
            if lb:
                self.dropobs = np.where(self.pi < lb)[0]
            elif self.pslb:
                self.dropobs = np.where(self.pi < self.pslb)[0]
            n_trimmed = self.dropobs.shape[0]
            print(f"{n_trimmed} observations trimmed")
            ifv = self.ifvals[~self.dropobs, :]
        else:  # no trimming - neyman take the wheel
            ifv = self.ifvals
        # influence functions for current computation
        ifvalsC = ifv[:, 0]
        ifvalsC = ifvalsC.reshape(ifvalsC.shape[0], 1)
        ifvalsT = np.delete(ifv, 0, axis=1)
        # causal contrasts
        self.causal_contrasts = ifvalsT - ifvalsC
        # marginal means and SE
        self.ATEs = self.causal_contrasts.mean(axis=0)
        self.SEs = np.sqrt(
            self.causal_contrasts.var(axis=0) / self.causal_contrasts.shape[0]
        )
        sumtab = np.c_[
            self.ATEs,
            self.SEs,
            self.ATEs - critval * self.SEs,
            self.ATEs + critval * self.SEs,
        ]
        # make a pretty table with labels
        self.sumtab = pd.DataFrame(
            data=sumtab,
            index=[
                f"Treat level {self.w_levels[k]} - Treat level {self.w_levels[0]}"
                for k in range(1, self.K)
            ],
            columns=["ATE", "SE", "95% CI-LB", "95% CI-UB"],
        )
        print(self.sumtab)
