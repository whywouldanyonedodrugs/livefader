# /opt/livefader/src/train_model.py
import os, asyncio, logging, joblib
from datetime import datetime, timezone
import asyncpg
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn import __version__ as sk_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("train_model")

MODEL_OUT = os.getenv("WINPROB_MODEL_PATH", "win_probability_model.pkl")

# === Top-level function (picklable) ===
def hour_to_sin_cos(x: np.ndarray) -> np.ndarray:
    """Encode hour-of-day as sin/cos on [0,24)."""
    hour = x.reshape(-1, 1).astype(float)
    sin = np.sin(2 * np.pi * hour / 24.0)
    cos = np.cos(2 * np.pi * hour / 24.0)
    return np.concatenate([sin, cos], axis=1)

# Raw features expected in DB (we’ll create missing with 0.0)
RAW_FEATURES = [
    "rsi_at_entry",
    "adx_at_entry",
    "price_boom_pct_at_entry",
    "price_slowdown_pct_at_entry",
    "vwap_dev_pct_at_entry",
    "ema_spread_pct_at_entry",
    "is_ema_crossed_down_at_entry",
    "eth_macdhist_at_entry",
    "day_of_week_at_entry",
    "hour_of_day_at_entry",
    "ret_30d_at_entry",
    "listing_age_days_at_entry",
]

TARGET_COL = "is_win"

def _fetch_env():
    load_dotenv()
    return dict(
        PG_DSN=os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/livedb"),
        MIN_TRADES=int(os.getenv("TRAIN_MIN_TRADES", "200")),
        C_L1=float(os.getenv("TRAIN_L1_C", "0.5")),
        CALIB_METHOD=os.getenv("CALIB_METHOD", "auto"),  # auto|isotonic|sigmoid
        SEED=int(os.getenv("SEED", "42")),
    )

async def _load_df(pg_dsn: str) -> pd.DataFrame:
    LOG.info("Fetching clean trade history from the database...")
    conn = await asyncpg.connect(pg_dsn)
    try:
        rows = await conn.fetch("""
            SELECT *
            FROM positions
            WHERE status='CLOSED' AND pnl IS NOT NULL
            ORDER BY id
        """)
        if not rows:
            raise RuntimeError("No closed trades with PnL found.")
        df = pd.DataFrame([dict(r) for r in rows])

        # Target
        pnl_num = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        df[TARGET_COL] = (pnl_num > 0).astype(int)

        # Ensure RAW_FEATURES exist; create missing with zeros
        for col in RAW_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        # Coerce numerics safely on feature columns only
        numeric_like = [
            f for f in RAW_FEATURES
            if f not in ("day_of_week_at_entry", "hour_of_day_at_entry")
        ]
        for col in numeric_like:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Robust day/hour (ints in valid ranges)
        df["day_of_week_at_entry"] = (
            pd.to_numeric(df["day_of_week_at_entry"], errors="coerce")
            .fillna(0).clip(0, 6).astype(int)
        )
        df["hour_of_day_at_entry"] = (
            pd.to_numeric(df["hour_of_day_at_entry"], errors="coerce")
            .fillna(0).clip(0, 23).astype(int)
        )

        # Fill ema_spread if you have components and spread is zero
        if "ema_fast_at_entry" in df.columns and "ema_slow_at_entry" in df.columns:
            ef = pd.to_numeric(df["ema_fast_at_entry"], errors="coerce")
            es = pd.to_numeric(df["ema_slow_at_entry"], errors="coerce")
            spread = np.where(es > 0, (ef - es) / es, 0.0)
            mask = (pd.to_numeric(df["ema_spread_pct_at_entry"], errors="coerce").fillna(0.0) == 0.0) & np.isfinite(spread)
            df.loc[mask, "ema_spread_pct_at_entry"] = spread[mask]

        LOG.info("Loaded %d trades for training.", len(df))
        return df
    finally:
        await conn.close()

def _build_pipeline(C_L1: float, calib_method: str, n_samples: int, seed: int) -> Pipeline:
    # cyclical hour encoder via top-level function (picklable)
    hour_enc = Pipeline(steps=[
        ("cyc", FunctionTransformer(hour_to_sin_cos, validate=True, feature_names_out="one-to-one"))
    ])

    # OneHotEncoder version compatibility: sparse_output (>=1.2) vs sparse (<1.2)
    major, minor = map(int, sk_version.split(".")[:2])
    if (major, minor) >= (1, 2):
        weekday_enc = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
    else:
        weekday_enc = OneHotEncoder(drop=None, sparse=False, handle_unknown="ignore")

    numeric_cols = [
        "rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_dev_pct_at_entry","ema_spread_pct_at_entry","is_ema_crossed_down_at_entry",
        "eth_macdhist_at_entry","ret_30d_at_entry","listing_age_days_at_entry"
    ]

    pre = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("weekday_1hot", weekday_enc, ["day_of_week_at_entry"]),
        ("hour_sin_cos", hour_enc, ["hour_of_day_at_entry"]),
    ], remainder="drop")

    base = LogisticRegression(
        penalty="l1", solver="liblinear", C=C_L1, max_iter=4000, random_state=seed
    )

    # Calibration choice from docs: isotonic needs many samples; else sigmoid/Platt
    if calib_method == "auto":
        method = "isotonic" if n_samples >= 1000 else "sigmoid"
    else:
        method = calib_method

    calibrated = CalibratedClassifierCV(estimator=base, method=method, cv=5)
    return Pipeline(steps=[("pre", pre), ("clf", calibrated)])

def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series):
    p = model.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return brier, auc

async def main():
    cfg = _fetch_env()
    df = await _load_df(cfg["PG_DSN"])
    if len(df) < cfg["MIN_TRADES"]:
        raise RuntimeError(f"Too few trades for training: {len(df)} < {cfg['MIN_TRADES']}")

    X = df[RAW_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    model = _build_pipeline(cfg["C_L1"], cfg["CALIB_METHOD"], len(df), cfg["SEED"])
    model.fit(X, y)

    brier, auc = _evaluate(model, X, y)
    LOG.info("In-sample Brier=%.5f  AUC=%.4f  N=%d", brier, auc, len(df))
    if auc < 0.52:
        LOG.warning("AUC is low; treat probabilities as weak for filtering/sizing until we validate OOS.")

    joblib.dump(
        dict(pipeline=model, features=RAW_FEATURES, trained_at=datetime.now(timezone.utc).isoformat()),
        MODEL_OUT
    )
    LOG.info("Saved calibrated model to %s", MODEL_OUT)

if __name__ == "__main__":
    asyncio.run(main())
