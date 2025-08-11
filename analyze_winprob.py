# /opt/livefader/src/analyze_winprob.py
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers ---

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    df = pd.DataFrame({"y": y_true, "p": y_prob, "bin": idx})
    g = df.groupby("bin", as_index=False).agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean"))
    g["w"] = g["n"] / g["n"].sum()
    ece = float((g["w"] * (g["y_rate"] - g["p_mean"]).abs()).sum())
    return g, ece

def make_calibration_plot(calib_df: pd.DataFrame, out_png: str, title="Reliability diagram"):
    plt.figure(figsize=(6,6))
    x = calib_df["p_mean"].values
    y = calib_df["y_rate"].values
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(x, y, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed win rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def decile_lift(df: pd.DataFrame, proba_col: str, label_col: str):
    # high→low deciles
    df = df.copy()
    df["decile"] = pd.qcut(df[proba_col].rank(method="first"), 10, labels=False, duplicates="drop")
    # rank high probs to decile 9
    mx = df["decile"].max()
    df["decile"] = mx - df["decile"]
    base = df[label_col].mean()
    t = df.groupby("decile").agg(
        n=("decile","size"),
        p_mean=(proba_col,"mean"),
        win_rate=(label_col,"mean")
    ).reset_index().sort_values("decile", ascending=False)
    t["lift"] = t["win_rate"] / base if base > 0 else np.nan
    return t, float(base)

def safe_bucket(series: pd.Series, kind: str):
    s = pd.to_numeric(series, errors="coerce")
    if kind == "sign":
        return pd.Series(np.where(s >= 0, "nonneg", "neg"), index=series.index)
    if kind == "q5":
        try:
            return pd.qcut(s, 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        except ValueError:
            return pd.Series(["UNK"]*len(series), index=series.index)
    return pd.Series(["UNK"]*len(series), index=series.index)

def bucket_kpis(df_pos: pd.DataFrame, out_csv: str):
    need = ["pnl","y","oof_proba","eth_macdhist_at_entry","vwap_stack_frac_at_entry","funding_rate_at_entry","oi_1h_change_at_entry"]
    for c in need:
        if c not in df_pos.columns:
            df_pos[c] = np.nan

    out_rows = []

    # ETH MACD hist (sign + quintiles)
    df_pos["eth_sign"] = safe_bucket(df_pos["eth_macdhist_at_entry"], "sign")
    df_pos["eth_q"] = safe_bucket(df_pos["eth_macdhist_at_entry"], "q5")
    # VWAP-stack frac
    df_pos["vwap_q"] = safe_bucket(df_pos["vwap_stack_frac_at_entry"], "q5")
    # Funding sign
    df_pos["funding_sign"] = safe_bucket(df_pos["funding_rate_at_entry"], "sign")
    # OI delta quintiles
    df_pos["oi_q"] = safe_bucket(df_pos["oi_1h_change_at_entry"], "q5")

    def agg_block(by):
        g = df_pos.groupby(by).agg(
            n=("y","size"),
            win_rate=("y","mean"),
            avg_pnl=("pnl","mean"),
            med_pnl=("pnl","median"),
            mean_proba=("oof_proba","mean")
        ).reset_index()
        g["group"] = by if isinstance(by, str) else "+".join(by)
        return g

    blocks = [
        agg_block("eth_sign"),
        agg_block("eth_q"),
        agg_block("vwap_q"),
        agg_block("funding_sign"),
        agg_block("oi_q"),
    ]
    res = pd.concat(blocks, ignore_index=True)
    res.to_csv(out_csv, index=False)
    return res

# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="Path to winprob_oof.csv")
    ap.add_argument("--trades", required=False, default=None, help="Path to trades.csv (must include id, pnl, *_at_entry features)")
    ap.add_argument("--outdir", required=False, default="results", help="Output directory")
    ap.add_argument("--bins", type=int, default=10, help="Calibration bins")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    oof = pd.read_csv(args.oof)
    if "position_id" not in oof.columns:
        oof["position_id"] = np.arange(len(oof))  # best-effort fallback

    # Basic summaries
    oof["y"] = pd.to_numeric(oof["y"], errors="coerce").fillna(0).astype(int)
    oof["oof_proba"] = pd.to_numeric(oof["oof_proba"], errors="coerce").clip(0,1).fillna(0.0)

    # Calibration (reliability diagram) & ECE
    calib_df, ece = expected_calibration_error(oof["y"].values, oof["oof_proba"].values, n_bins=args.bins)
    calib_csv = os.path.join(args.outdir, "winprob_calibration.csv")
    calib_df.to_csv(calib_csv, index=False)
    make_calibration_plot(calib_df, os.path.join(args.outdir, "winprob_calibration.png"),
                          title=f"Reliability (ECE={ece:.3f})")

    # Decile lift
    deciles, base_rate = decile_lift(oof, "oof_proba", "y")
    deciles.to_csv(os.path.join(args.outdir, "winprob_deciles.csv"), index=False)

    # Bucketed KPIs (requires trades.csv to get features & pnl)
    if args.trades:
        trades = pd.read_csv(args.trades)
        # normalize column names
        trades.columns = [c.strip() for c in trades.columns]
        id_col = "id" if "id" in trades.columns else ("position_id" if "position_id" in trades.columns else None)
        if id_col is None:
            print("[WARN] trades file has no id/position_id column; skipping bucket KPIs.")
        else:
            pos = trades.rename(columns={id_col: "position_id"})
            pos = pos.merge(oof[["position_id","oof_proba","y"]], on="position_id", how="inner")
            # Coerce PnL numeric
            pos["pnl"] = pd.to_numeric(pos.get("pnl", 0.0), errors="coerce").fillna(0.0)
            out_csv = os.path.join(args.outdir, "winprob_buckets.csv")
            res = bucket_kpis(pos, out_csv)
            print(f"[OK] Saved bucketed KPIs → {out_csv}")
    else:
        print("[INFO] No trades CSV supplied; skipped bucketed KPIs.")

    print(f"[OK] Saved: {calib_csv}, winprob_calibration.png, winprob_deciles.csv")

if __name__ == "__main__":
    main()
