# research/modeling/pipeline.py
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def _hour_cyc(X: np.ndarray) -> np.ndarray:
    # X is (n,1) hour; return sin/cos columns
    h = X.astype(float).ravel()
    return np.c_[np.sin(2*np.pi*h/24.0), np.cos(2*np.pi*h/24.0)]

def build_feature_lists(df_cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    cont = [
        "rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_z_at_entry","ema_spread_pct_at_entry","eth_macdhist_at_entry",
        "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry","vwap_stack_slope_pph_at_entry",
        "funding_last_at_entry","oi_last_at_entry","oi_delta_pct_win","listing_age_days_at_entry"
    ]
    cont = [c for c in cont if c in df_cols]

    cyc_hour = "hour_of_day_at_entry" if "hour_of_day_at_entry" in df_cols else None
    cat = ["day_of_week_at_entry", "session_tag_at_entry", "vwap_consolidated_at_entry"]
    cat = [c for c in cat if c in df_cols]
    return cont, cat, [cyc_hour] if cyc_hour else []

def build_pipeline(df: pd.DataFrame):
    cont, cat, cyc = build_feature_lists(df.columns.tolist())

    transformers = []
    if cont:
        transformers.append(("cont", StandardScaler(with_mean=True, with_std=True), cont))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if cyc:
        cyc_tf = Pipeline([
            ("sel", "passthrough"),  # the column is selected by name via ColumnTransformer
            ("tr", FunctionTransformer(_hour_cyc, validate=False)),
        ])
        transformers.append(("cyc", cyc_tf, cyc))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    base = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, max_iter=200, class_weight="balanced"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", base),
    ])
    return pipe, cont, cat, cyc
