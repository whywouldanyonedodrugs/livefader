# analyze_trades.py
import argparse
from pathlib import Path
import os
import asyncio
from datetime import datetime, timezone
import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from sklearn.metrics import brier_score_loss

# Optional: statsmodels quick logit, gated by --quick-logit
try:
    import statsmodels.api as sm  # noqa
except Exception:
    sm = None

# Optional live-time adapter (best-effort; analyzer will fall back if missing)
try:
    from live.winprob_loader import WinProbScorer  # type: ignore
except Exception:
    WinProbScorer = None  # analyzer will continue without it

RESULTS_DIR = Path("results")

# stub to allow unpickling if a pickle references __main__.ModelBundle
class ModelBundle:
    pass

def _hour_cyc(x):
    import numpy as _np
    hour = _np.asarray(x).reshape(-1, 1).astype(float)
    sin = _np.sin(2 * _np.pi * hour / 24.0)
    cos = _np.cos(2 * _np.pi * hour / 24.0)
    return _np.concatenate([sin, cos], axis=1)

# alias used by some old pickles
def hour_to_sin_cos(x):
    return _hour_cyc(x)


def _unwrap_estimator_and_features(obj) -> Tuple[object, Optional[List[str]]]:
    """
    Return (estimator_like, feature_names_or_None) from a variety of bundle shapes:
      - dicts: {'calibrator': CalibratedClassifierCV, 'feature_names': [...]}
               {'pipeline': sklearn Pipeline, 'feature_names': [...]}
      - dataclass/objects: .calibrator, .feature_names  OR .pipeline
      - raw sklearn estimator/pipeline (has .predict_proba)
      - statsmodels results (has .predict and .model.exog_names)
    """
    # dict bundle
    if isinstance(obj, dict):
        feats = obj.get("feature_names") or obj.get("features") or None
        if "calibrator" in obj:
            return obj["calibrator"], feats
        if "pipeline" in obj:
            return obj["pipeline"], feats
        # otherwise fall through to return raw dict (won't score)

    # dataclass / class bundle
    if hasattr(obj, "calibrator"):
        feats = getattr(obj, "feature_names", None)
        return getattr(obj, "calibrator"), feats
    if hasattr(obj, "pipeline"):
        feats = getattr(obj, "feature_names", None)
        return getattr(obj, "pipeline"), feats

    # raw estimator or statsmodels results
    return obj, None


def _statsmodels_predict(results_obj, df: pd.DataFrame) -> np.ndarray:
    """
    Build exog to match statsmodels' expected columns & constant, then call .predict().
    """
    if not hasattr(results_obj, "predict") or not hasattr(results_obj, "model"):
        raise TypeError("Provided statsmodels object is missing .predict or .model.")
    exog_names = list(getattr(results_obj.model, "exog_names", []))  # includes 'const' if present
    names_wo_const = [n for n in exog_names if n != "const"]
    X = df.reindex(columns=names_wo_const, fill_value=0.0).astype(float)
    if sm is None:
        raise RuntimeError("statsmodels is not available to add constant/predict.")
    X = sm.add_constant(X, prepend=True, has_constant="add")
    p = results_obj.predict(X)
    return np.asarray(p, dtype=float)


# -----------------------
# Research model features
# -----------------------
_FEATURES_FOR_MODEL: List[str] = [
    "rsi_at_entry",
    "adx_at_entry",
    "price_boom_pct_at_entry",
    "price_slowdown_pct_at_entry",
    "vwap_z_at_entry",
    "ema_spread_pct_at_entry",
    "eth_macdhist_at_entry",
    "vwap_stack_frac_at_entry",
    "vwap_stack_expansion_pct_at_entry",
    "vwap_stack_slope_pph_at_entry",
    "day_of_week_at_entry",
    "hour_of_day_at_entry",
]

def _prep_X_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature frame the sklearn research pipeline expects by name.
    """
    needed = _FEATURES_FOR_MODEL
    X = pd.DataFrame(index=df.index)
    for c in needed:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    X["day_of_week_at_entry"] = X["day_of_week_at_entry"].fillna(0).clip(0,6).astype(int)
    X["hour_of_day_at_entry"] = X["hour_of_day_at_entry"].fillna(0).clip(0,23).astype(int)
    return X.fillna(0.0)

def _load_model_object(model_path: str):
    """
    Load the serialized object (bundle or estimator/results) and return it as-is.
    """
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Expected a file, got a directory: {p}")
    return joblib.load(p)

def _build_feat_dict(row: pd.Series) -> dict:
    ema_slow = float(row.get("ema_slow_at_entry", 0.0))
    ema_fast = float(row.get("ema_fast_at_entry", 0.0))
    ema_spread = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
    return {
        "rsi_at_entry": float(row.get("rsi_at_entry", 0.0)),
        "adx_at_entry": float(row.get("adx_at_entry", 0.0)),
        "price_boom_pct_at_entry": float(row.get("price_boom_pct_at_entry", 0.0)),
        "price_slowdown_pct_at_entry": float(row.get("price_slowdown_pct_at_entry", 0.0)),
        "vwap_z_at_entry": float(row.get("vwap_z_at_entry", 0.0)),
        "ema_spread_pct_at_entry": float(row.get("ema_spread_pct_at_entry", ema_spread)),
        "day_of_week_at_entry": int(row.get("day_of_week_at_entry", 0)) % 7,
        "hour_of_day_at_entry": int(row.get("hour_of_day_at_entry", 0)) % 24,
        "eth_macdhist_at_entry": float(row.get("eth_macdhist_at_entry", 0.0)),
        "vwap_stack_frac_at_entry": float(row.get("vwap_stack_frac_at_entry", 0.0)),
        "vwap_stack_expansion_pct_at_entry": float(row.get("vwap_stack_expansion_pct_at_entry", 0.0)),
        "vwap_stack_slope_pph_at_entry": float(row.get("vwap_stack_slope_pph_at_entry", 0.0)),
    }

def _init_scorer(model_path: str):
    """Try constructing the live adapter with either positional or named path argument."""
    if WinProbScorer is None:
        raise RuntimeError("WinProbScorer not importable")
    try:
        return WinProbScorer(model_path)      # new-style ctor
    except TypeError:
        return WinProbScorer(path=model_path) # fallback signature


def score_with_model(df: pd.DataFrame, model_path: Optional[str]) -> Optional[np.ndarray]:
    if not model_path:
        print("No model path provided; skipping model scoring.")
        return None

    # Path A: use the same adapter the live bot uses (best if present)
    try:
        scorer = _init_scorer(model_path)  # will raise if adapter not available
        preds = []
        for _, row in df.iterrows():
            feat = _build_feat_dict(row)
            try:
                preds.append(float(scorer.score(feat)))
            except Exception:
                preds.append(np.nan)
        arr = np.asarray(preds, dtype=float)
        if np.isfinite(arr).any():
            return arr
    except Exception as e:
        print(f"WinProbScorer path failed: {e}")

    # Path B: load the serialized estimator/bundle directly
    try:
        raw = _load_model_object(model_path)
        est, feat_names = _unwrap_estimator_and_features(raw)

        # scikit-learn estimator with probabilities (pipeline or calibrator)
        if hasattr(est, "predict_proba"):
            if feat_names:
                # Use the exact training feature list; fill missing with 0.0
                X = df.reindex(columns=list(feat_names), fill_value=0.0).astype(float)
            else:
                # Fallback to a generic prep if feature_names were not saved
                X = _prep_X_for_model(df)
            return est.predict_proba(X)[:, 1].astype(float)

        # statsmodels result: build exog by exog_names (+ add constant)
        if hasattr(est, "predict") and hasattr(est, "model") and hasattr(est.model, "exog_names"):
            p = _statsmodels_predict(est, df)
            return np.asarray(p, dtype=float)

        print("Loaded object cannot produce probabilities (no predict_proba or statsmodels predict).")
        return None

    except Exception as e:
        print(f"Direct path failed: {e}")
        return None


# -----------------------
# Pretty printing helpers
# -----------------------
def header(txt: str):
    bar = "=" * 72
    print(f"\n{bar}\n {txt}\n{bar}")

def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"

def _fmt2(x: float) -> str:
    return f"{x:.2f}"

# --------------------------------
# Basic feature engineering / flags
# --------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Net PnL (fees-aware)
    if "fees_paid" in df.columns:
        df["fees_paid"] = pd.to_numeric(df["fees_paid"], errors="coerce").fillna(0.0)
        df["net_pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0) - df["fees_paid"]
    else:
        df["net_pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    # Boolean outcome
    df["is_win"] = df["net_pnl"] > 0

    # Safe numerics used later
    for c in ("rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
              "ema_fast_at_entry","ema_slow_at_entry","vwap_z_at_entry",
              "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry",
              "vwap_stack_slope_pph_at_entry","holding_minutes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # EMA-derived helpers
    if {"entry_price", "ema_fast_at_entry", "ema_slow_at_entry"}.issubset(df.columns):
        df["price_minus_ema_fast"] = df["entry_price"] - df["ema_fast_at_entry"]
        df["price_minus_ema_slow"] = df["entry_price"] - df["ema_slow_at_entry"]
        df["pct_to_ema_fast"] = df["price_minus_ema_fast"] / df["entry_price"]
        df["pct_to_ema_slow"] = df["price_minus_ema_slow"] / df["entry_price"]
        df["ema_spread_abs"] = df["ema_fast_at_entry"] - df["ema_slow_at_entry"]
        df["ema_spread_pct"] = df["ema_spread_abs"] / df["ema_slow_at_entry"]

    # Historical alias
    if "vwap_dev_pct_at_entry" in df.columns and "pct_to_vwap" not in df.columns:
        df.rename(columns={"vwap_dev_pct_at_entry": "pct_to_vwap"}, inplace=True)

    # Holding time buckets
    if "holding_minutes" in df.columns:
        df["holding_minutes"] = df["holding_minutes"].fillna(0.0)
        df["holding_hours"] = df["holding_minutes"] / 60.0
        df["holding_bucket"] = pd.cut(
            df["holding_hours"],
            bins=[0, 1, 2, 4, 8, 24, np.inf],
            labels=["<1 h", "1-2 h", "2-4 h", "4-8 h", "8-24 h", ">24 h"],
        )
    else:
        df["holding_hours"] = np.nan
        df["holding_bucket"] = np.nan

    return df

# --------------------
# KPI & performance
# --------------------
def _profit_factor(pnl: pd.Series) -> float:
    g = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(g / l) if l > 0 else np.inf

def top_line_kpis(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    wr = df["is_win"].mean() if n else np.nan
    exp_trade = df["net_pnl"].mean()
    pf = _profit_factor(df["net_pnl"])
    hold_hours = df["holding_hours"].mean()
    wins = df.loc[df["net_pnl"] > 0, "net_pnl"].mean()
    losses = df.loc[df["net_pnl"] < 0, "net_pnl"].mean()
    return pd.DataFrame([{
        "trades": n,
        "win_rate": wr,
        "expectancy": exp_trade,
        "profit_factor": pf,
        "avg_hold_hours": hold_hours,
        "avg_win": wins,
        "avg_loss": losses,
    }])

# -------------------------------------
# Calibration, ECE and reliability bins
# -------------------------------------
def _ece(y: np.ndarray, p: np.ndarray, m_bins: int = 10) -> Tuple[pd.DataFrame, float]:
    y = y.astype(int)
    p = p.astype(float)
    bins = np.linspace(0.0, 1.0, m_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, m_bins - 1)
    rows = []
    ece = 0.0
    for b in range(m_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            rows.append(dict(bin=b, lower=bins[b], upper=bins[b+1], n=0,
                             avg_p=np.nan, win_rate=np.nan, abs_gap=np.nan))
            continue
        avg_p = float(p[mask].mean())
        win_rate = float(y[mask].mean())
        abs_gap = abs(avg_p - win_rate)
        rows.append(dict(bin=b, lower=bins[b], upper=bins[b+1], n=n_b,
                         avg_p=avg_p, win_rate=win_rate, abs_gap=abs_gap))
        ece += (n_b / len(y)) * abs_gap
    return pd.DataFrame(rows), float(ece)

def _brier(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    base = float(np.mean((y - y.mean()) ** 2))  # constant-forecast baseline
    score = float(np.mean((y - p) ** 2))        # Brier score (lower is better)
    return score, base

def decile_lift(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    try:
        df["decile"] = pd.qcut(df["p"], q=10, labels=False, duplicates="drop")
    except Exception:
        df["decile"] = pd.qcut(df["p"].rank(method="first"), q=10, labels=False, duplicates="drop")
    g = (df.groupby("decile")
           .agg(n=("y", "size"), avg_p=("p", "mean"), win_rate=("y", "mean"))
           .reset_index()
           .sort_values("decile", ascending=False))
    g["lift_vs_base"] = g["win_rate"] - df["y"].mean()
    return g

# -----------------------
# Descriptive breakdowns
# -----------------------
def describe_continuous(df, col, win_flag="is_win"):
    header(f"{col} – wins vs. losses")
    grp = df.groupby(win_flag)[col].describe()[["count", "mean", "std", "25%", "50%", "75%"]]
    print(grp.to_string())
    if df[col].notna().sum() > 10:
        try:
            r, p = stats.pointbiserialr(df[win_flag].fillna(False), df[col].fillna(0))
            print(f"\nPoint-biserial corr with win flag: r={r:.3f}  p={p:.4f}")
        except Exception:
            pass

def describe_categorical(df, col, win_flag="is_win"):
    header(f"{col} – win rate by category")
    tab = pd.crosstab(df[col], df[win_flag], margins=True)
    if True in tab.columns:
        tab["win_rate_%"] = 100 * tab[True] / tab["All"]
    print(tab.to_string())
    if len(tab) > 1 and tab.shape[0] > 2 and tab.shape[1] > 2:
        try:
            chi2 = stats.chi2_contingency(tab.iloc[:-1, :-1])[0]
            n = tab["All"].iloc[-1]
            k = min(tab.shape[0] - 1, tab.shape[1] - 1)
            cramers_v = np.sqrt(chi2 / (n * k)) if n and k else np.nan
            print(f"\nCramér’s V vs. win flag: {cramers_v:.3f}")
        except Exception:
            pass

def _safe_qcut(s: pd.Series, q: int):
    return pd.qcut(s, q, labels=False, duplicates="drop")

def _save_table(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")

def _vwap_quintile_tables(df: pd.DataFrame) -> dict:
    out = {}
    if "vwap_stack_frac_at_entry" in df.columns:
        s = pd.to_numeric(df["vwap_stack_frac_at_entry"], errors="coerce")
        if s.notna().sum() >= 5:
            df["vwap_frac_q"] = _safe_qcut(s, 5)
            g = (df.groupby("vwap_frac_q")
                   .agg(n=("id","size"),
                        win_rate=("is_win","mean"),
                        mean_net_pnl=("net_pnl","mean"),
                        median_net_pnl=("net_pnl","median"))
                   .reset_index())
            out["vwap_frac_quintiles"] = g
    if "vwap_stack_expansion_pct_at_entry" in df.columns:
        s = pd.to_numeric(df["vwap_stack_expansion_pct_at_entry"], errors="coerce")
        if s.notna().sum() >= 5:
            df["vwap_exp_q"] = _safe_qcut(s, 5)
            g = (df.groupby("vwap_exp_q")
                   .agg(n=("id","size"),
                        win_rate=("is_win","mean"),
                        mean_net_pnl=("net_pnl","mean"),
                        median_net_pnl=("net_pnl","median"))
                   .reset_index())
            out["vwap_expansion_quintiles"] = g
    if "vwap_stack_slope_pph_at_entry" in df.columns:
        s = pd.to_numeric(df["vwap_stack_slope_pph_at_entry"], errors="coerce")
        if s.notna().sum() >= 5:
            df["vwap_slope_q"] = _safe_qcut(s, 5)
            g = (df.groupby("vwap_slope_q")
                   .agg(n=("id","size"),
                        win_rate=("is_win","mean"),
                        mean_net_pnl=("net_pnl","mean"),
                        median_net_pnl=("net_pnl","median"))
                   .reset_index())
            out["vwap_slope_quintiles"] = g
    return out

# -----------------------------
# Optional quick logistic check
# -----------------------------
def quick_logit(df: pd.DataFrame, features, enabled: bool):
    if not enabled:
        return
    if sm is None:
        warnings.warn("statsmodels not installed – skipping quick logistic regression.")
        return
    clean_df = df.dropna(subset=features + ["is_win"])
    if len(clean_df) < 25:
        print("Not enough data for logistic regression.")
        return
    header("Quick logistic regression (wins=1) — optional sanity check")
    X = sm.add_constant(clean_df[features])
    y = clean_df["is_win"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())
    except Exception as e:
        print(f"Logistic regression failed: {e}")

# --------------------------
# Markdown report generation
# --------------------------
def _write_markdown_report(title: str,
                           kpis: pd.DataFrame,
                           brier_tuple: Optional[Tuple[float, float]],
                           ece_val: Optional[float],
                           rel_path: Optional[Path],
                           lift_path: Optional[Path]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"report_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"

    lines = [f"# {title}",
             "",
             "## Top-line KPIs",
             "",
             kpis.to_string(index=False),
             ""]

    if brier_tuple is not None and ece_val is not None:
        brier, base = brier_tuple
        lines += [
            "## Probability Quality",
            "",
            f"- Brier score (lower is better): **{brier:.5f}**  (constant baseline: {base:.5f})",
            f"- Expected Calibration Error (ECE): **{ece_val:.5f}**",
            f"- Reliability table: `{rel_path.name}`" if rel_path else "",
            f"- Decile lift table: `{lift_path.name}`" if lift_path else "",
            ""
        ]

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join([ln for ln in lines if ln is not None]))
    print(f"[saved] {out}")
    return out

# -------------------------
# Telegram delivery (file)
# -------------------------
async def send_telegram_report(report_file_path: str):
    from dotenv import load_dotenv
    import telegram  # python-telegram-bot
    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not found in .env file. Skipping report send.")
        return
    try:
        bot = telegram.Bot(token=token)
        print(f"Sending report file {report_file_path} to Telegram chat {chat_id}...")
        with open(report_file_path, 'rb') as document:
            await bot.send_document(
                chat_id=chat_id,
                document=document,
                filename=os.path.basename(report_file_path),
                caption=f"LiveFader Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            )
        print("Telegram report sent successfully.")
    except FileNotFoundError:
        print(f"Report file not found at {report_file_path}.")
    except Exception as e:
        print(f"Failed to send report to Telegram: {e}")

# -----------------
# Main entry point
# -----------------
def main():
    ap = argparse.ArgumentParser(description="Trade analytics console report (model-aware)")
    ap.add_argument("--file", default="trades.csv", help="CSV of closed trades")
    ap.add_argument("--equity-file", default="equity_data.csv", help="CSV of equity snapshots (optional)")
    ap.add_argument("--model-path", default="win_probability_model.pkl", help="Path to calibrated research model bundle")
    ap.add_argument("--bins", type=int, default=10, help="Bins for reliability/ECE")
    ap.add_argument("--quick-logit", action="store_true", help="Run optional statsmodels Logit sanity check")
    ap.add_argument("--send-report", help="Path to a file to be sent via Telegram (skip analysis)")
    args = ap.parse_args()

    if args.send_report:
        asyncio.run(send_telegram_report(args.send_report))
        return

    trade_file_path = Path(args.file)
    if not trade_file_path.exists():
        print(f"Trade file not found: {trade_file_path}")
        return

    # Load & engineer
    df = pd.read_csv(trade_file_path)
    if df.empty:
        print("No rows in trade file.")
        return
    df = feature_engineering(df)

    # Top-line KPIs
    header("TOP-LINE KPIs")
    kpis = top_line_kpis(df)
    print(
        kpis.assign(
            win_rate=kpis["win_rate"].map(_fmt_pct),
            expectancy=kpis["expectancy"].map(_fmt2),
            profit_factor=kpis["profit_factor"].map(lambda x: "inf" if np.isinf(x) else _fmt2(x))
        ).to_string(index=False)
    )

    # Score with research model (if provided) and evaluate probability quality
    rel_path = None
    lift_path = None
    ece_val = None
    brier_tuple = None

    preds = score_with_model(df, args.model_path)
    if preds is not None and np.isfinite(preds).any():
        df["winprob_pred"] = preds
        y = df["is_win"].astype(int).values
        p = np.nan_to_num(df["winprob_pred"].values, nan=np.mean(y))

        header("CALIBRATION & LIFT (research model)")
        rel, ece_val = _ece(y, p, m_bins=max(5, args.bins))
        print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        brier_tuple = _brier(y, p)
        print(f"\nBrier={brier_tuple[0]:.5f}  (constant baseline={brier_tuple[1]:.5f})")

        lift = decile_lift(y, p)
        print("\nDecile lift (sorted high→low p):")
        print(lift.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        _save_table(rel, "reliability_table")
        _save_table(lift, "decile_lift")
        rel_path = RESULTS_DIR / "reliability_table.csv"
        lift_path = RESULTS_DIR / "decile_lift.csv"
    else:
        print("\nNo model predictions available; skipping calibration/lift sections.")

    # Regime performance
    if "market_regime_at_entry" in df.columns:
        header("REGIME-SPECIFIC PERFORMANCE")
        regime = (df.groupby("market_regime_at_entry")
                    .agg(total_trades=("net_pnl", "size"),
                         win_rate=("is_win", "mean"),
                         avg_pnl=("net_pnl", "mean"),
                         profit_factor=("net_pnl", _profit_factor))
                    .reset_index())
        regime["win_rate_%"] = 100.0 * regime["win_rate"]
        print(regime[["market_regime_at_entry", "total_trades", "win_rate_%", "avg_pnl", "profit_factor"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        _save_table(regime, "regime_kpis")

    # VWAP-stack quintiles
    header("VWAP-STACK QUINTILES (frac, expansion, slope) — win rate & net PnL")
    for name, tab in _vwap_quintile_tables(df).items():
        tab2 = tab.copy()
        tab2["win_rate_%"] = 100 * tab2["win_rate"].astype(float)
        key_col = [c for c in ("vwap_frac_q", "vwap_exp_q", "vwap_slope_q") if c in tab2.columns][0]
        print(f"\n{name}")
        print(tab2[[key_col, "n", "win_rate_%", "mean_net_pnl", "median_net_pnl"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        _save_table(tab2, name)

    # Descriptive bands (concise)
    if "rsi_at_entry" in df.columns:
        df["rsi_band"] = pd.cut(df["rsi_at_entry"], bins=[0, 50, 60, 70, 80, 100],
                                labels=["<50", "50-60", "60-70", "70-80", ">80"])
        describe_categorical(df, "rsi_band")
    for cont in ("pct_to_ema_fast", "ema_spread_pct", "pct_to_vwap", "vwap_z_at_entry"):
        if cont in df.columns:
            describe_continuous(df, cont)
    if "vwap_consolidated_at_entry" in df.columns:
        describe_categorical(df, "vwap_consolidated_at_entry")
    for cat in ("session_tag_at_entry", "holding_bucket", "day_of_week_at_entry", "hour_of_day_at_entry"):
        if cat in df.columns:
            describe_categorical(df, cat)

    # Optional quick logit (sanity check)
    cont_cols = [c for c in df.select_dtypes(include="number").columns if c not in ["is_win", "pnl", "net_pnl"]]
    cors = {c: stats.pointbiserialr(df["is_win"], df[c])[0]
            for c in cont_cols if df[c].notna().sum() > 5 and np.std(df[c]) > 0}
    top10 = pd.Series(cors).abs().sort_values(ascending=False).head(10).index.tolist()
    quick_logit(df, [c for c in top10 if c not in ("pnl_pct", "mae_usd", "mfe_usd")], args.quick_logit)

    # Markdown summary
    report_file = _write_markdown_report(
        title="LiveFader Trade Report",
        kpis=top_line_kpis(df),
        brier_tuple=brier_tuple,
        ece_val=ece_val,
        rel_path=rel_path,
        lift_path=lift_path
    )
    print("\nDone.")

if __name__ == "__main__":
    main()
