"""Native Python ACIC benchmark data generators.

The helpers carry the ACIC covariate matrices and parameter grids inside the
package and expose stable 1-based setting/replication conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr, ndtri


ACIC2016_N_SETTINGS = 77
ACIC2016_N_REPLICATIONS = 100
ACIC2017_N_SETTINGS = 32
ACIC2017_N_REPLICATIONS = 250

_DATA_PACKAGE = "aipyw.data.acic"


@dataclass
class ACICSample:
    """Container returned by ACIC benchmark simulators.

    Attributes:
        y: Observed outcome vector.
        a: Binary treatment vector.
        X: Numeric design matrix suitable for AIPyW.
        x: Original covariate frame.
        setting: 1-based parameter-grid row.
        replication: 1-based replication index.
        parameters: Parameter row as a dictionary.
        y0: Realized control potential outcome, when available.
        y1: Realized treated potential outcome, when available.
        mu0: Conditional mean under control, when available.
        mu1: Conditional mean under treatment, when available.
        e: Propensity score, when available.
        alpha: Individual treatment-effect vector, when available.
    """

    y: np.ndarray
    a: np.ndarray
    X: np.ndarray
    x: pd.DataFrame
    setting: int
    replication: int
    parameters: dict
    y0: np.ndarray | None = None
    y1: np.ndarray | None = None
    mu0: np.ndarray | None = None
    mu1: np.ndarray | None = None
    e: np.ndarray | None = None
    alpha: np.ndarray | None = None

    @property
    def tau(self) -> np.ndarray:
        """Individual treatment effects on the outcome scale."""
        if self.alpha is not None:
            return self.alpha
        if self.mu0 is not None and self.mu1 is not None:
            return self.mu1 - self.mu0
        if self.y0 is not None and self.y1 is not None:
            return self.y1 - self.y0
        raise AttributeError("this sample does not contain treatment-effect truth")

    @property
    def true_ate(self) -> float:
        return float(np.mean(self.tau))

    @property
    def true_att(self) -> float:
        treated = self.a == 1
        if not np.any(treated):
            return float("nan")
        return float(np.mean(self.tau[treated]))

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(y, a, X)`` in the convention used by ``aipyw.dgp``."""
        return self.y, self.a, self.X

    def to_frame(self) -> pd.DataFrame:
        """Return a compact observed-data frame with available truth columns."""
        out = self.x.copy()
        out["z"] = self.a
        out["y"] = self.y
        for name in ("y0", "y1", "mu0", "mu1", "e", "alpha"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


def _data_path(name: str):
    return resources.files(_DATA_PACKAGE).joinpath(name)


def _read_csv(name: str) -> pd.DataFrame:
    with resources.as_file(_data_path(name)) as path:
        return pd.read_csv(path)


def load_acic2016_input() -> pd.DataFrame:
    """Load the ACIC 2016 covariate matrix."""
    return _read_csv("input_2016.csv")


def load_acic2016_parameters() -> pd.DataFrame:
    """Load the 77-row ACIC 2016 parameter grid."""
    return _read_csv("parameters_2016.csv")


def load_acic2017_input() -> pd.DataFrame:
    """Load the ACIC 2017 covariate matrix."""
    return _read_csv("input_2017.csv")


def load_acic2017_parameters() -> pd.DataFrame:
    """Load the 32-row ACIC 2017 parameter grid."""
    out = _read_csv("parameters_2017.csv")
    if "error" not in out.columns and "errors" in out.columns:
        out = out.rename(columns={"errors": "error"})
    return out


def load_acic2017_transformed() -> pd.DataFrame:
    """Load the transformed covariates used by the 2017 DGP."""
    return _read_csv("transformed_data_2017.csv")


def _validate_index(value: int, upper: int, name: str) -> int:
    value = int(value)
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}")
    return value


def _resolve_parameters(
    parameters: int | Mapping | pd.Series | pd.DataFrame,
    grid: pd.DataFrame,
    *,
    name: str,
) -> tuple[int, dict]:
    if isinstance(parameters, (int, np.integer)):
        setting = _validate_index(int(parameters), len(grid), name)
        return setting, grid.iloc[setting - 1].to_dict()
    if isinstance(parameters, pd.DataFrame):
        if len(parameters) != 1:
            raise ValueError("parameter data frame must have exactly one row")
        parameters = parameters.iloc[0]
    out = dict(parameters)
    return -1, out


def _rep_seed(year: int, setting: int, replication: int, random_seed: int | None) -> int:
    if random_seed is not None:
        return int(random_seed)
    seq = np.random.SeedSequence([year, int(setting), int(replication)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


@lru_cache(maxsize=2)
def _prepared_design(year: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    x = load_acic2016_input() if year == 2016 else load_acic2017_input()
    design = pd.get_dummies(x, drop_first=True, dtype=float)
    arr = design.to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)
    scale[scale == 0] = 1.0
    z = (arr - mean) / scale
    return x, design, z


def _feature_bank(
    z: np.ndarray,
    rng: np.random.Generator,
    model: str,
    *,
    n_terms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = z.shape[1]
    cols = rng.choice(n_features, size=min(n_terms, n_features), replace=False)
    base = z[:, cols]
    if model == "linear":
        return base, cols, np.full(base.shape[1], "linear", dtype=object)
    if model == "polynomial":
        pieces = [base]
        kinds = ["linear"] * base.shape[1]
        pieces.append(base[:, : max(2, base.shape[1] // 2)] ** 2 - 1.0)
        kinds.extend(["quadratic"] * pieces[-1].shape[1])
        if base.shape[1] >= 4:
            interactions = [
                (base[:, j] * base[:, j + 1])[:, None]
                for j in range(0, min(base.shape[1] - 1, 8), 2)
            ]
            pieces.extend(interactions)
            kinds.extend(["interaction"] * len(interactions))
        return np.column_stack(pieces), cols, np.array(kinds, dtype=object)
    if model == "step":
        thresholds = rng.normal(0.0, 0.5, size=base.shape[1])
        steps = (base > thresholds).astype(float) - 0.5
        pieces = [steps]
        kinds = ["step"] * steps.shape[1]
        if base.shape[1] >= 4:
            hinges = np.maximum(base[:, : base.shape[1] // 2] - thresholds[: base.shape[1] // 2], 0.0)
            pieces.append(hinges)
            kinds.extend(["hinge"] * hinges.shape[1])
        return np.column_stack(pieces), cols, np.array(kinds, dtype=object)
    if model == "exponential":
        clipped = np.clip(base, -2.5, 2.5)
        exp_terms = np.exp(0.35 * clipped) - np.exp(0.35 * clipped).mean(axis=0)
        pieces = [base[:, : max(2, base.shape[1] // 2)], exp_terms]
        kinds = ["linear"] * pieces[0].shape[1] + ["exponential"] * exp_terms.shape[1]
        return np.column_stack(pieces), cols, np.array(kinds, dtype=object)
    raise ValueError(f"unknown ACIC model kind {model!r}")


def _weighted_sum(features: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    weights = rng.standard_t(df=4, size=features.shape[1])
    weights = weights / max(np.linalg.norm(weights), 1e-12)
    score = features @ weights
    score = (score - score.mean()) / (score.std() + 1e-12)
    return scale * score, weights


def _overlap_mask(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_features = z.shape[1]
    j, k = rng.choice(n_features, size=2, replace=False)
    lo_j, hi_j = np.quantile(z[:, j], [0.15, 0.55])
    lo_k, hi_k = np.quantile(z[:, k], [0.45, 0.85])
    return (z[:, j] >= lo_j) & (z[:, j] <= hi_j) & (z[:, k] >= lo_k) & (z[:, k] <= hi_k)


def simulate_acic2016(
    setting: int = 1,
    replication: int = 1,
    *,
    random_seed: int | None = None,
    standardize: bool = False,
) -> ACICSample:
    """Generate a native-Python ACIC 2016 benchmark sample.

    The simulator keeps the ACIC 2016 covariates, 77-row parameter grid,
    100-replication convention, and main difficulty knobs.  It returns full
    potential-outcome truth for scalar ATE/ATT benchmarking.
    """
    grid = load_acic2016_parameters()
    setting, pars = _resolve_parameters(setting, grid, name="setting")
    replication = _validate_index(replication, ACIC2016_N_REPLICATIONS, "replication")
    rng = np.random.default_rng(_rep_seed(2016, setting, replication, random_seed))
    x, design, z_design = _prepared_design(2016)

    model_trt = str(pars["model.trt"])
    model_rsp = str(pars["model.rsp"])
    root_trt = float(pars["root.trt"])
    overlap = str(pars["overlap.trt"])
    alignment = float(pars["alignment"])
    te_hetero = str(pars["te.hetero"])

    trt_features, _, _ = _feature_bank(z_design, rng, model_trt, n_terms=18)
    trt_score, trt_weights = _weighted_sum(trt_features, rng, scale=1.8)
    e = expit(np.log(root_trt / (1 - root_trt)) + trt_score)
    if overlap in {"one-term", "two-term"}:
        e = np.where(_overlap_mask(z_design, rng), np.minimum(e, 0.02), e)
    if overlap == "two-term":
        e = np.where(_overlap_mask(z_design, rng), np.minimum(e, 0.02), e)
    e = np.clip(e, 0.01, 0.99)
    a = rng.binomial(1, e).astype(int)

    rsp_features, _, _ = _feature_bank(z_design, rng, model_rsp, n_terms=22)
    rsp_weights = rng.standard_t(df=3, size=rsp_features.shape[1])
    rsp_weights = rsp_weights / max(np.linalg.norm(rsp_weights), 1e-12)
    shared = min(len(trt_weights), len(rsp_weights), trt_features.shape[1], rsp_features.shape[1])
    if shared > 0:
        aligned = np.zeros_like(rsp_weights)
        aligned[:shared] = np.sign(rsp_weights[:shared]) * np.abs(trt_weights[:shared])
        rsp_weights = alignment * aligned + (1 - alignment) * rsp_weights
        rsp_weights = rsp_weights / max(np.linalg.norm(rsp_weights), 1e-12)
    mu_base = rsp_features @ rsp_weights
    mu_base = 2.0 * (mu_base - mu_base.mean()) / (mu_base.std() + 1e-12)

    true_att = 0.75 + (1.0 / 6.0) * rng.standard_t(df=3)
    if te_hetero == "none":
        tau = np.full(len(x), true_att)
    else:
        strength = 0.45 if te_hetero == "med" else 0.9
        h_features, _, _ = _feature_bank(z_design, rng, "polynomial", n_terms=10)
        h_score, _ = _weighted_sum(h_features, rng, scale=strength)
        tau = true_att + h_score
        treated = a == 1
        if np.any(treated):
            tau = tau + (true_att - tau[treated].mean())

    mu0 = mu_base
    mu1 = mu0 + tau
    y0 = rng.normal(mu0, 1.0)
    y1 = rng.normal(mu1, 1.0)
    y = np.where(a == 1, y1, y0)
    X = z_design if standardize else design.to_numpy(dtype=float)

    return ACICSample(
        y=y,
        a=a,
        X=X,
        x=x.copy(),
        setting=setting,
        replication=replication,
        parameters=pars,
        y0=y0,
        y1=y1,
        mu0=mu0,
        mu1=mu1,
        e=e,
    )


_FACTOR_LEVELS = {
    "x_3": ["leq_0", "gt_0"],
    "x_10": ["leq_0", "gt_0"],
    "x_14": ["leq_0", "gt_0"],
    "x_15": ["leq_0", "gt_0"],
    "x_21": list("ABCDEFGHIJKLMNOP"),
    "x_24": list("ABCDE"),
}


def _factor_code(series: pd.Series, levels: list[str]) -> np.ndarray:
    dtype = pd.CategoricalDtype(categories=levels, ordered=True)
    codes = series.astype(dtype).cat.codes.to_numpy()
    if np.any(codes < 0):
        bad = sorted(set(series[codes < 0].astype(str)))
        raise ValueError(f"unexpected factor levels in {series.name}: {bad}")
    return codes + 1


def simulate_acic2017(
    setting: int = 1,
    replication: int = 1,
    *,
    random_seed: int | None = None,
    standardize: bool = False,
) -> ACICSample:
    """Generate a native-Python ACIC 2017 benchmark sample.

    This follows the compact ACIC 2017 DGP formula and returns the individual
    treatment-effect vector ``alpha`` used for CATE/ITE-flavored scoring.
    """
    grid = load_acic2017_parameters()
    setting, pars = _resolve_parameters(setting, grid, name="setting")
    replication = _validate_index(replication, ACIC2017_N_REPLICATIONS, "replication")
    rng = np.random.default_rng(_rep_seed(2017, setting, replication, random_seed))
    x_raw, design, z_design = _prepared_design(2017)
    x = load_acic2017_transformed()

    error = str(pars["error"])
    magnitude = float(pars["magnitude"])
    noise = float(pars["noise"])
    confounding = float(pars["confounding"])

    effect_size = 1 / 3 if magnitude <= 0 else 2
    beta0, beta1 = (0.0, 0.5) if confounding <= 0 else (-1.0, 3.0)
    snr = 0.25 if noise <= 0 else 1.25
    reffectv = 0.1 if error == "group_corr" else 0.0

    x10 = _factor_code(x["x_10"], _FACTOR_LEVELS["x_10"])
    x14 = _factor_code(x["x_14"], _FACTOR_LEVELS["x_14"])
    x15 = _factor_code(x["x_15"], _FACTOR_LEVELS["x_15"])
    x21 = _factor_code(x["x_21"], _FACTOR_LEVELS["x_21"])

    p_arg = x["x_1"].to_numpy() + x["x_43"].to_numpy() + 0.3 * (2 - x10)
    e = 1 / (1 + np.exp(beta0 + beta1 * p_arg))
    mu = -np.sin(ndtri(e)) + x["x_43"].to_numpy()
    alpha = effect_size * (
        ((x["x_3"].to_numpy() == "leq_0") & (x["x_24"].to_numpy() == "B")).astype(float)
        + (2 - x14)
        - (2 - x15)
    )

    linpart = mu + e * alpha
    sigma_y = snr * np.std(linpart, ddof=1)

    if error == "nonadditive":
        b = np.sqrt(sigma_y**2 + np.var(linpart, ddof=1))
        c = np.mean(linpart)
        muvec1 = (mu + alpha - c) / (1.25 * b)
        muvec0 = (mu - c) / (1.25 * b)
        denom = np.sqrt(sigma_y**2 / (1.25 * b) ** 2 + 1)
        alpha_out = 13 * ndtr(muvec1 / denom) - 13 * ndtr(muvec0 / denom)
    else:
        b = c = None
        alpha_out = alpha

    het = np.linspace(0.4, 1.4, 16) if error == "heteroskedastic" else np.ones(16)
    a = rng.binomial(1, e, size=len(x)).astype(int)
    y_temp = mu + het[x21 - 1] * (1 - reffectv) * sigma_y * rng.normal(size=len(x))
    y_obs = y_temp + a * alpha
    y_cf = y_temp + (1 - a) * alpha

    ref = reffectv * sigma_y * rng.normal(size=len(_FACTOR_LEVELS["x_21"]))
    y_obs = y_obs + ref[x21 - 1]
    y_cf = y_cf + ref[x21 - 1]

    if error == "nonadditive":
        eps = rng.normal(0, 0.01, size=len(x))
        y_obs = 13 * ndtr((y_obs - c) / (1.25 * b)) - 6 + eps
        y_cf = 13 * ndtr((y_cf - c) / (1.25 * b)) - 6 + eps

    X = z_design if standardize else design.to_numpy(dtype=float)
    return ACICSample(
        y=y_obs,
        a=a,
        X=X,
        x=x_raw.copy(),
        setting=setting,
        replication=replication,
        parameters=pars,
        e=e,
        alpha=alpha_out,
    )


def iter_acic2016(
    *,
    settings: Iterable[int] | None = None,
    replications: Iterable[int] | None = None,
    standardize: bool = False,
) -> Iterable[ACICSample]:
    """Iterate over the ACIC 2016 benchmark grid."""
    settings = range(1, ACIC2016_N_SETTINGS + 1) if settings is None else settings
    replications = range(1, ACIC2016_N_REPLICATIONS + 1) if replications is None else replications
    for setting in settings:
        for replication in replications:
            yield simulate_acic2016(setting, replication, standardize=standardize)


def iter_acic2017(
    *,
    settings: Iterable[int] | None = None,
    replications: Iterable[int] | None = None,
    standardize: bool = False,
) -> Iterable[ACICSample]:
    """Iterate over the ACIC 2017 benchmark grid."""
    settings = range(1, ACIC2017_N_SETTINGS + 1) if settings is None else settings
    replications = range(1, ACIC2017_N_REPLICATIONS + 1) if replications is None else replications
    for setting in settings:
        for replication in replications:
            yield simulate_acic2017(setting, replication, standardize=standardize)
