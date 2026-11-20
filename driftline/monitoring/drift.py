"""Distribution drift: a reference profile is captured at training time and stored with the
model; live or batch data is scored against it with PSI (binned) and a two-sample KS test.

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats as sps

EPS = 1e-4
N_BINS = 10
TOP_K_CATEGORIES = 30
KS_SAMPLE = 5_000


@dataclass
class FeatureProfile:
    kind: str  # "numeric" | "categorical"
    edges: list[float] = field(default_factory=list)       # numeric: interior bin edges
    categories: list[int] = field(default_factory=list)    # categorical: top-K, rest = "other"
    expected: list[float] = field(default_factory=list)    # proportion per bin (+ missing)
    sample: list[float] = field(default_factory=list)      # numeric: reference sample for KS


def _numeric_props(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    missing = np.isnan(values)
    counts = np.bincount(np.searchsorted(edges, values[~missing], side="right"),
                         minlength=len(edges) + 1)
    return np.append(counts, missing.sum()) / max(len(values), 1)


def _categorical_props(values: np.ndarray, categories: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values)
    present = values[~missing]
    counts = [np.count_nonzero(present == c) for c in categories]
    other = len(present) - sum(counts)
    return np.array(counts + [other, missing.sum()], dtype=np.float64) / max(len(values), 1)


def build_profile(df: pl.DataFrame, categorical: list[str], seed: int = 7) -> dict[str, FeatureProfile]:
    rng = np.random.default_rng(seed)
    profile = {}
    for col in df.columns:
        if col in categorical:
            top = (df.get_column(col).drop_nulls().value_counts(sort=True)
                   .head(TOP_K_CATEGORIES).get_column(col).to_list())
            cats = [int(c) for c in top]
            values = df.get_column(col).cast(pl.Float64).fill_null(np.nan).to_numpy()
            profile[col] = FeatureProfile(
                "categorical", categories=cats,
                expected=_categorical_props(values, cats).tolist())
        else:
            values = df.get_column(col).cast(pl.Float64).fill_null(np.nan).to_numpy()
            finite = values[~np.isnan(values)]
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1)[1:-1]))
            sample = rng.choice(finite, size=min(KS_SAMPLE, len(finite)), replace=False)
            profile[col] = FeatureProfile(
                "numeric", edges=edges.tolist(),
                expected=_numeric_props(values, edges).tolist(), sample=sample.tolist())
    return profile


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected), EPS, None)
    a = np.clip(np.asarray(actual), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks_pvalue: float | None
    drifted: bool


def score_drift(profile: dict[str, FeatureProfile], current: pl.DataFrame,
                psi_threshold: float = 0.2) -> list[DriftResult]:
    results = []
    for col, prof in profile.items():
        if col not in current.columns:
            continue
        if prof.kind == "categorical":
            values = current.get_column(col).cast(pl.Float64).fill_null(np.nan).to_numpy()
            actual = _categorical_props(values, prof.categories)
            ks_p = None
        else:
            values = current.get_column(col).cast(pl.Float64).fill_null(np.nan).to_numpy()
            actual = _numeric_props(values, np.asarray(prof.edges))
            finite = values[~np.isnan(values)]
            ks_p = float(sps.ks_2samp(prof.sample, finite).pvalue) if len(finite) else None
        value = psi(np.asarray(prof.expected), actual)
        results.append(DriftResult(col, round(value, 4), ks_p, value > psi_threshold))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def profile_to_json(profile: dict[str, FeatureProfile]) -> dict:
    return {k: vars(v) for k, v in profile.items()}


def profile_from_json(data: dict) -> dict[str, FeatureProfile]:
    return {k: FeatureProfile(**v) for k, v in data.items()}
