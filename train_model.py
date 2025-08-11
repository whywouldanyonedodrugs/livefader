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
from sklearn.model_selection import StratifiedKFold
from sklearn import __version__ as sk_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger("train_model")

MODEL_OUT = os.getenv("WINPROB_MODEL_PATH", "win_probability_model.pkl")
OOF_CSV   = os.getenv("WINPROB_OOF_PATH",   "winprob_oof.csv")

# === Top-level function (picklable) ===
def hour_to_sin_cos(x: np.ndarray) -> np.ndarray:
    """Encode hour-of-day as sin/cos on [0,24)."""
    hour = x.reshape(-1, 1).astype(float)
    sin = np.sin(2 * np.pi * hour / 24.0)
    cos = np.cos(2 * np.pi * hour / 24.0)
    return np.concatenate([sin, cos], axis=1)

RAW_FEATURES = [
    "rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
    "vwap_dev_pct_at_entry","ema_spread_pct_at_entry","is_ema_crossed_down_at_entry",
    "eth_macdhist_at_entry","day_of_week_at_entry","hour_of_day_at_entry",
    "ret_30d_at_entry","listing_age_days_at_entry",
]
TARGET_COL = "is_win"

def _fetch_env():
    load_dotenv()
    return dict(
        PG_DSN=os.getenv("PG_DSN", "postgresql://livefader:livepw@localhost:5432/livedb"),
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

        pnl_num = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        df[TARGET_COL] = (pnl_num > 0).astype(int)

        # Ensure features exist
        for col in RAW_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        # Numerics
        numeric_like = [f for f in RAW_FEATURES if f not in ("day_of_week_at_entry","hour_of_day_at_entry")]
        for col in numeric_like:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Day/hour as valid ints
        df["day_of_week_at_entry"] = pd.to_numeric(df["day_of_week_at_entry"], errors="coerce").fillna(0).clip(0,6).astype(int)
        df["hour_of_day_at_entry"] = pd.to_numeric(df["hour_of_day_at_entry"], errors="coerce").fillna(0).clip(0,23).astype(int)

        # Backfill EMA spread if components exist
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

def _weekday_encoder():
    major, minor = map(int, sk_version.split(".")[:2])
    if (major, minor) >= (1, 2):
        return OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
    return OneHotEncoder(drop=None, sparse=False, handle_unknown="ignore")

def _preprocessor():
    hour_enc = Pipeline(steps=[("cyc", FunctionTransformer(hour_to_sin_cos, validate=True, feature_names_out="one-to-one"))])
    weekday_enc = _weekday_encoder()
    numeric_cols = [
        "rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_dev_pct_at_entry","ema_spread_pct_at_entry","is_ema_crossed_down_at_entry",
        "eth_macdhist_at_entry","ret_30d_at_entry","listing_age_days_at_entry"
    ]
    return ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("weekday_1hot", weekday_enc, ["day_of_week_at_entry"]),
        ("hour_sin_cos", hour_enc, ["hour_of_day_at_entry"]),
    ], remainder="drop")

def _calib_method(choice: str, n_samples: int) -> str:
    if choice == "auto":
        return "isotonic" if n_samples >= 1000 else "sigmoid"  # isotonic needs lots of data; sigmoid (Platt) is safer when data is scarce. :contentReference[oaicite:3]{index=3}
    return choice

def _build_pipeline(C_L1: float, calib_method: str, n_samples: int, seed: int) -> Pipeline:
    pre = _preprocessor()
    base = LogisticRegression(penalty="l1", solver="liblinear", C=C_L1, max_iter=4000, random_state=seed)
    method = _calib_method(calib_method, n_samples)
    calibrated = CalibratedClassifierCV(estimator=base, method=method, cv=5)
    pipe = Pipeline(steps=[("pre", pre), ("clf", calibrated)])
    pipe._chosen_calibration = method  # for printing later
    return pipe

def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series):
    p = model.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return brier, auc, p

def _oof_cv(pipeline_builder, X: pd.DataFrame, y: pd.Series, C_L1: float, calib_method: str, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    folds = np.zeros(len(y), dtype=int)
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        pipe = pipeline_builder(C_L1, calib_method, len(tr), seed + i)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:,1]
        folds[va] = i
    brier = brier_score_loss(y, oof)
    auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else float("nan")
    return oof, folds, brier, auc

def _coef_report(X: pd.DataFrame, y: pd.Series, C_L1: float, seed: int, top_k: int = 12):
    """Fit an uncalibrated L1 LR (same preprocess) to report non-zero coefficients."""
    pre = _preprocessor()
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=C_L1, max_iter=4000, random_state=seed)
    pipe = Pipeline(steps=[("pre", pre), ("lr", lr)])
    pipe.fit(X, y)
    # feature names after preprocessor
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["lr"].coef_.ravel()
    nz = np.flatnonzero(coefs)
    pairs = [(names[i], float(coefs[i])) for i in nz]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:top_k], len(nz), len(coefs)

async def main():
    cfg = _fetch_env()
    df = await _load_df(cfg["PG_DSN"])
    if len(df) < cfg["MIN_TRADES"]:
        raise RuntimeError(f"Too few trades for training: {len(df)} < {cfg['MIN_TRADES']}")

    X = df[RAW_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    # Class balance & baseline Brier (constant predictor = base win-rate)
    pos_rate = float(y.mean())
    brier_baseline = brier_score_loss(y, np.full_like(y, fill_value=pos_rate, dtype=float))
    LOG.info("Class balance: win-rate=%.2f%%  (pos_rate=%.4f)", pos_rate*100, pos_rate)

    model = _build_pipeline(cfg["C_L1"], cfg["CALIB_METHOD"], len(df), cfg["SEED"])
    model.fit(X, y)

    brier_in, auc_in, p_in = _evaluate(model, X, y)
    LOG.info("In-sample Brier=%.5f  (baseline=%.5f)  AUC=%.4f  N=%d", brier_in, brier_baseline, auc_in, len(df))
    LOG.info("Calibration used: %s", getattr(model, "_chosen_calibration", "sigmoid"))

    # OOF CV diagnostics
    oof, folds, brier_oof, auc_oof = _oof_cv(_build_pipeline, X, y, cfg["C_L1"], cfg["CALIB_METHOD"], cfg["SEED"])
    pd.DataFrame({"oof_proba": oof, "y": y.values, "fold": folds}).to_csv(OOF_CSV, index=False)
    LOG.info("OOF 5-fold  Brier=%.5f  (baseline=%.5f)  AUC=%.4f  → saved %s", brier_oof, brier_baseline, auc_oof, OOF_CSV)

    # Coefficient report from uncalibrated L1 LR (directional signal)
    top, nnz, dim = _coef_report(X, y, cfg["C_L1"], cfg["SEED"], top_k=12)
    LOG.info("L1 LR non-zero coefficients: %d of %d features after encoding", nnz, dim)
    for name, w in top:
        LOG.info("  coef %-40s %+ .6f", name, w)

    # Inversion hint
    if auc_in < 0.5:
        LOG.warning("AUC_in %.3f < 0.5 → model ranks backwards in-sample (flip would be %.3f).", auc_in, 1.0 - auc_in)
    if auc_oof < 0.5:
        LOG.warning("AUC_OOF %.3f < 0.5 → OUT-OF-FOLD also backwards; do NOT use for sizing; investigate labels/features.", auc_oof)

    joblib.dump(dict(pipeline=model, features=RAW_FEATURES, trained_at=datetime.now(timezone.utc).isoformat()), MODEL_OUT)
    LOG.info("Saved calibrated model to %s", MODEL_OUT)

if __name__ == "__main__":
    asyncio.run(main())
