# research/modeling/pipeline.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from research.modeling.feature_mask import load_disabled_features, filter_feature_list

# ---- tiny helpers for cyclical encoding ----
def _hour_cyc(X: pd.Series | np.ndarray) -> np.ndarray:
    h = np.asarray(X).astype(float).reshape(-1, 1)
    sin = np.sin(2 * np.pi * h / 24.0)
    cos = np.cos(2 * np.pi * h / 24.0)
    return np.hstack([sin, cos])

def _dow_onehot(X: pd.Series | np.ndarray) -> np.ndarray:
    d = np.asarray(X).astype(int).reshape(-1, 1)
    out = np.zeros((d.shape[0], 7), dtype=float)
    m = (d >= 0) & (d < 7)
    out[np.arange(d.shape[0])[m.ravel()], d[m].ravel()] = 1.0
    return out

def build_pipeline(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str], list[str]]:
    """
    Build the LR(L1) + encoding pipeline.
    Returns (pipeline, cont_features, cat_features, cyc_features_names)
    """
    # Candidate columns present in dataset
    cont = [c for c in [
        "rsi_at_entry", "adx_at_entry",
        "price_boom_pct_at_entry", "price_slowdown_pct_at_entry",
        "vwap_z_at_entry", "ema_spread_pct_at_entry",
        "eth_macdhist_at_entry",
        "vwap_stack_frac_at_entry", "vwap_stack_expansion_pct_at_entry", "vwap_stack_slope_pph_at_entry",
    ] if c in df.columns]

    cat  = []  # (we don't have true categoricals yet)
    cycH = ["hour_of_day_at_entry"] if "hour_of_day_at_entry" in df.columns else []
    cycD = ["day_of_week_at_entry"] if "day_of_week_at_entry" in df.columns else []

    # Apply feature mask
    disabled = load_disabled_features()
    cont = filter_feature_list(cont, disabled)
    cycH = filter_feature_list(cycH, disabled)
    cycD = filter_feature_list(cycD, disabled)

    # Column transformers
    transformers = []
    if cont:
        transformers.append(("cont", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ]), cont))

    if cycH:
        transformers.append(("hour", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("cyc", FunctionTransformer(_hour_cyc, validate=False)),
        ]), cycH))

    if cycD:
        transformers.append(("dow", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", FunctionTransformer(_dow_onehot, validate=False)),
        ]), cycD))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    # Sparse-ish L1 logistic (robust, interpretable), balanced if needed
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.5,
        class_weight=None, random_state=42, max_iter=200
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    # names after encoding are opaque (x0..), but cont/cyc lists help with ablation/reporting
    used_cyc = []
    if cycH: used_cyc += ["hour_sin", "hour_cos"]
    if cycD: used_cyc += [f"dow_{i}" for i in range(7)]

    return pipe, cont, cat, used_cyc
