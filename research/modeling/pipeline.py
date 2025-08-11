# research/modeling/pipeline.py
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def _hour_cyc(X) -> np.ndarray:
    """
    Robust cyclical encoding for hour-of-day.
    Accepts DataFrame/Series/ndarray of shape (n,) or (n,1).
    NaNs -> 0, values wrapped into [0,24).
    Returns array of shape (n, 2): [sin(hour), cos(hour)].
    """
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    h = arr[:, 0]
    h = np.nan_to_num(h, nan=0.0)
    h = np.mod(h, 24.0)
    angle = 2.0 * np.pi * (h / 24.0)
    return np.c_[np.sin(angle), np.cos(angle)]

def _make_ohe():
    """Build OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _non_empty(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Keep only columns that have at least one non-NaN entry."""
    keep = []
    for c in cols:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
    return keep

def build_feature_lists(df_cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    cont = [
        "rsi_at_entry","adx_at_entry",
        "price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_z_at_entry","ema_spread_pct_at_entry","eth_macdhist_at_entry",
        "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry","vwap_stack_slope_pph_at_entry",
        "funding_last_at_entry","oi_last_at_entry","oi_delta_pct_win",
        "listing_age_days_at_entry",
    ]
    cat = ["day_of_week_at_entry", "session_tag_at_entry", "vwap_consolidated_at_entry"]
    cyc = ["hour_of_day_at_entry"]

    # presence filter only (we’ll apply non-empty filter in build_pipeline with df)
    cont = [c for c in cont if c in df_cols]
    cat = [c for c in cat if c in df_cols]
    cyc = [c for c in cyc if c in df_cols]
    return cont, cat, cyc

def build_pipeline(df: pd.DataFrame):
    cont, cat, cyc = build_feature_lists(df.columns.tolist())

    # NEW: drop NaN-only columns to avoid SimpleImputer warnings
    cont = _non_empty(df, cont)
    cat  = _non_empty(df, cat)
    cyc  = _non_empty(df, cyc)

    transformers = []

    if cont:
        cont_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ])
        transformers.append(("cont", cont_pipe, cont))

    if cat:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, cat))

    if cyc:
        cyc_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("hour_cyc", FunctionTransformer(_hour_cyc, validate=False)),
        ])
        transformers.append(("cyc", cyc_pipe, cyc))

    pre = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    base = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=200,
        class_weight="balanced",
        n_jobs=None,
        random_state=0,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", base),
    ])
    return pipe, cont, cat, cyc
