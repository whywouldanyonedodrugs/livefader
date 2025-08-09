"""
analysis_runner.py  (updated)
Reads a trades CSV and runs quant checks:
- VWAP distance and Z-score threshold grids
- EMA / Session / Hour / Consolidation gates
- 30d return threshold grid
- Payoff grid "what-if" for alternative TP/SL using MAE/MFE (in ATR units)
  * Recognizes TP exits as meeting the 1× ATR test via `exit_reason`
  * Applies a 5% tolerance on ATR multiples (MFE/MAE comparisons)
  * Optionally uses counterfactual columns: cf_would_hit_tp_*x_atr, cf_would_hit_sl_*x_atr

Outputs a plain-text report to stdout.

Assumes the CSV has at least:
  - pnl (numeric)  OR both gross_pnl and fees to compute net pnl
  - pct_to_vwap (absolute distance), vwap_z_at_entry (signed z)
  - ret_30d_at_entry
  - is_ema_crossed_down_at_entry (0/1)  [optional]
  - session_tag_at_entry in {ASIA, EUROPE, US}
  - hour_of_day_at_entry in 0..23
  - vwap_consolidated_at_entry (0/1)    [optional]
  - mae_over_atr, mfe_over_atr (float)  [for TP/SL what-if]
  - exit_reason / exit_type (optional, used to recognize TP hits at 1× ATR)
  - optional counterfactuals:
      cf_would_hit_tp_1x_atr, cf_would_hit_tp_1_2x_atr, ..., cf_would_hit_sl_2_5x_atr, etc.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Tuple, Dict, List
import math

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Utilities

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _ensure_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pnl" in df.columns:
        return df
    if "gross_pnl" in df.columns and "fees" in df.columns:
        df["pnl"] = df["gross_pnl"] - df["fees"]
        return df
    raise ValueError("CSV must contain 'pnl' or both 'gross_pnl' and 'fees'.")

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~95%
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2*n)
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)
    lower = (centre - half) / denom
    upper = (centre + half) / denom
    return lower, upper

def diff_in_prop_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float, float]:
    """Return (z, p_two_sided, p_one_sided) for difference in proportions test."""
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan, np.nan)
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    if se == 0:
        return (np.nan, np.nan, np.nan)
    z = (p1 - p2) / se
    # normal cdf
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_two = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    p_one = 1 - cdf
    return z, p_two, p_one

def mde_wr(n_in: int, n_out: int, base_wr: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Approx minimal detectable absolute improvement in WR (in-gate vs out-gate)
    using a normal approximation: delta ≈ (z_α + z_β) * sqrt(p*(1-p)*(1/n1+1/n2)).
    """
    if n_in == 0 or n_out == 0:
        return np.nan
    # z-values
    z_alpha = 1.6448536269514722  # one-sided 5%
    z_beta  = 0.8416212335729143  # 80% power
    delta = (z_alpha + z_beta) * math.sqrt(base_wr*(1-base_wr)*(1/n_in + 1/n_out))
    return float(delta)

def fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{100*x:6.2f}%"

def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    n = len(df)
    wins = int((df["pnl"] > 0).sum())
    wr = wins / n if n else np.nan
    gp = df.loc[df["pnl"] > 0, "pnl"].sum()
    gl = -df.loc[df["pnl"] <= 0, "pnl"].sum()
    pf = (gp / gl) if gl > 0 else np.nan
    avg = df["pnl"].mean() if n else np.nan
    lwr, upr = wilson_ci(wins, n) if n else (np.nan, np.nan)
    return dict(n=n, wins=wins, win_rate=wr, wr_lwr=lwr, wr_upr=upr, pf=pf, avg_pnl=avg, gp=gp, gl=gl)

def what_if_exclude(df: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
    kept = df[mask].copy()
    return _metrics(kept)

# ──────────────────────────────────────────────────────────────────────────────
# Threshold grid helpers

def threshold_grid_abs(df: pd.DataFrame, col: str, thresholds: Iterable[float], keep_op: str = "<=") -> pd.DataFrame:
    rows = []
    x = df[col].abs()
    for t in thresholds:
        if keep_op == "<=":
            mask = x <= t
            rule = f"|{col}| ≤ {t}"
        else:
            mask = x >= t
            rule = f"|{col}| ≥ {t}"
        m = what_if_exclude(df, mask)
        m.update(dict(rule=rule, kept=int(mask.sum()), dropped=int((~mask).sum())))
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["pf","avg_pnl"], ascending=[False, False])

def threshold_grid_signed(df: pd.DataFrame, col: str, thresholds: Iterable[float], direction: str = ">=") -> pd.DataFrame:
    rows = []
    x = df[col]
    for t in thresholds:
        if direction == ">=":
            mask = x >= t
            rule = f"{col} ≥ {t}"
        else:
            mask = x <= t
            rule = f"{col} ≤ {t}"
        m = what_if_exclude(df, mask)
        m.update(dict(rule=rule, kept=int(mask.sum()), dropped=int((~mask).sum())))
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["pf","avg_pnl"], ascending=[False, False])

def binary_gate_test(df: pd.DataFrame, mask: pd.Series, label: str = "gate") -> pd.DataFrame:
    a = df[mask]
    b = df[~mask]
    ka, na = int((a["pnl"] > 0).sum()), len(a)
    kb, nb = int((b["pnl"] > 0).sum()), len(b)
    z, p_two, p_one = diff_in_prop_test(ka, na, kb, nb)
    base_wr = (df["pnl"] > 0).mean()
    row = dict(
        gate=label,
        in_n=na, in_wr=a["pnl"].gt(0).mean() if na else np.nan, in_pf=_metrics(a)["pf"],
        out_n=nb, out_wr=b["pnl"].gt(0).mean() if nb else np.nan, out_pf=_metrics(b)["pf"],
        wr_mde_abs=mde_wr(na, nb, base_wr),
        z=z, p_one_sided=p_one, p_two_sided=p_two,
    )
    return pd.DataFrame([row])

# ──────────────────────────────────────────────────────────────────────────────
# Payoff what-if using MAE/MFE in ATR units
#  - Recognizes TP exits for 1x tests via exit_reason (TP/TP1/TP_FINAL/etc.)
#  - 5% tolerance on thresholds
#  - Uses cf_* columns if present

TP_REASON_TOKENS = {
    "tp", "tp1", "tp2", "tp_final", "take_profit", "takeprofit", "take_profit_1", "take_profit_final"
}

def _normalize_reason(val: str) -> str:
    if not isinstance(val, str):
        return ""
    return val.strip().lower().replace(" ", "_")

def _has_cf_col(df: pd.DataFrame, kind: str, multiple: float) -> str | None:
    """
    kind: 'tp' or 'sl'
    multiple: e.g., 1.0, 1.2, 2.5
    Tries to find a column like:
      cf_would_hit_tp_1x_atr
      cf_would_hit_tp_1_2x_atr
      cf_would_hit_sl_2_5x_atr
    Returns the column name if found, else None.
    """
    base = f"cf_would_hit_{kind}_"
    # Try "1x" style
    s1 = f"{base}{multiple:g}x_atr"
    if s1 in df.columns:
        return s1
    # Try "1_2x" style
    mult_str = str(multiple).replace(".", "_")
    s2 = f"{base}{mult_str}x_atr"
    if s2 in df.columns:
        return s2
    # Try with possible trailing zeros stripped
    mult_str2 = f"{multiple:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    s3 = f"{base}{mult_str2}x_atr"
    if s3 in df.columns:
        return s3
    return None

def payoff_eval(df: pd.DataFrame, tp_grid: Iterable[float], sl_grid: Iterable[float], tol: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate alternative (TP, SL) pairs using MAE/MFE over ATR.
    Returns three tables: optimistic, pessimistic, midpoint.
    - optimistic: if MFE>=TP count as win; elif MAE>=SL loss; else 0 (unresolved)
    - pessimistic: if MAE>=SL loss; elif MFE>=TP win; else 0
    - midpoint: average of the two R outcomes

    Enhancements:
    - If 'exit_reason' indicates TP, treat TP>=1.0x as hit for thresholds <= 1.0x*(1+tol)
    - Apply 5% tolerance: effective_tp = tp*(1-tol), effective_sl = sl*(1-tol)
    - If cf_would_hit_tp_*x_atr / cf_would_hit_sl_*x_atr columns exist, use them
      to override comparisons at the matching thresholds.
    """
    required = {"mae_over_atr", "mfe_over_atr"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV lacks required columns for payoff sim: {required}")
    mae = df["mae_over_atr"].astype(float)
    mfe = df["mfe_over_atr"].astype(float)

    # Exit reason normalization
    exit_reason = None
    for c in ("exit_reason", "reason", "exit_type"):
        if c in df.columns:
            exit_reason = df[c].astype(str).map(_normalize_reason)
            break

    def hits_tp_via_reason(threshold: float) -> pd.Series:
        """True where threshold <= ~1.0x and reason indicates a TP exit."""
        if exit_reason is None:
            return pd.Series(False, index=df.index)
        # accept any threshold up to 1.0*(1+tol)
        if threshold <= 1.0*(1+tol):
            return exit_reason.isin(TP_REASON_TOKENS)
        return pd.Series(False, index=df.index)

    def eval_table(policy: str) -> pd.DataFrame:
        rows: List[Dict] = []
        for sl in sl_grid:
            for tp in tp_grid:
                eff_tp = tp * (1 - tol)
                eff_sl = sl * (1 - tol)
                reward = tp / sl if sl > 0 else np.nan

                # Optional counterfactual overrides (exact threshold matches)
                cf_tp_col = _has_cf_col(df, "tp", tp)
                cf_sl_col = _has_cf_col(df, "sl", sl)
                cf_tp = df[cf_tp_col].astype(bool) if cf_tp_col else None
                cf_sl = df[cf_sl_col].astype(bool) if cf_sl_col else None

                pnl_R = []
                wins = 0
                losses = 0
                unresolved = 0
                # Precompute reason-based TP hits for ~1x thresholds
                tp_hit_by_reason = hits_tp_via_reason(tp)

                for i, (a, f) in enumerate(zip(mae, mfe)):
                    # Decide hit/sl via (cf columns) or via MFE/MAE with tolerance
                    hit_tp = False
                    hit_sl = False
                    if cf_tp is not None:
                        hit_tp = bool(cf_tp.iloc[i])
                    else:
                        if not np.isnan(f) and f >= eff_tp:
                            hit_tp = True
                        # allow reason-based TP recognition near 1x
                        elif tp_hit_by_reason.iloc[i]:
                            hit_tp = True

                    if cf_sl is not None:
                        hit_sl = bool(cf_sl.iloc[i])
                    else:
                        if not np.isnan(a) and a >= eff_sl:
                            hit_sl = True

                    # Apply policy
                    if policy == "optimistic":
                        if hit_tp:
                            pnl_R.append(reward); wins += 1
                        elif hit_sl:
                            pnl_R.append(-1.0); losses += 1
                        else:
                            pnl_R.append(0.0); unresolved += 1
                    elif policy == "pessimistic":
                        if hit_sl:
                            pnl_R.append(-1.0); losses += 1
                        elif hit_tp:
                            pnl_R.append(reward); wins += 1
                        else:
                            pnl_R.append(0.0); unresolved += 1
                    else:
                        # midpoint: average of optimistic and pessimistic outcomes
                        if hit_tp and not hit_sl:
                            r_mid = reward
                        elif hit_sl and not hit_tp:
                            r_mid = -1.0
                        elif hit_tp and hit_sl:
                            # unknown order → average of extremes
                            r_mid = 0.5 * (reward + -1.0)
                        else:
                            r_mid = 0.0
                        pnl_R.append(r_mid)
                        if r_mid > 0:
                            wins += 1
                        elif r_mid < 0:
                            losses += 1
                        else:
                            unresolved += 1

                pnl_R = np.array(pnl_R, float)
                n = len(pnl_R)
                gp = pnl_R[pnl_R > 0].sum()
                gl = -pnl_R[pnl_R < 0].sum()
                pf = (gp / gl) if gl > 0 else np.nan
                wr = wins / (wins + losses) if (wins + losses) > 0 else np.nan
                rows.append(dict(
                    policy=policy, tp=tp, sl=sl, reward_R=reward,
                    n=n, wins=wins, losses=losses, unresolved=unresolved,
                    win_rate=wr, avg_R=float(np.nanmean(pnl_R)),
                    pf=pf
                ))
        return pd.DataFrame(rows).sort_values(["avg_R","pf","win_rate"], ascending=[False, False, False])

    return eval_table("optimistic"), eval_table("pessimistic"), eval_table("midpoint")

# ──────────────────────────────────────────────────────────────────────────────
# Runner

def run(path: str) -> None:
    df = pd.read_csv(path)
    df = _lower_cols(df)
    df = _ensure_pnl(df)

    # Harmonize obvious column aliases
    col_map = {
        "pct_to_vwap": ["pct_to_vwap", "pct_to_vwap_at_entry", "vwap_dev_pct_at_entry"],
        "vwap_z_at_entry": ["vwap_z_at_entry", "vwap_z_score_at_entry", "vwap_z"],
        "ret_30d_at_entry": ["ret_30d_at_entry", "ret_30d"],
        "is_ema_crossed_down_at_entry": ["is_ema_crossed_down_at_entry", "ema_down", "ema_cross_down"],
        "session_tag_at_entry": ["session_tag_at_entry", "session", "session_tag"],
        "hour_of_day_at_entry": ["hour_of_day_at_entry", "entry_hour"],
        "vwap_consolidated_at_entry": ["vwap_consolidated_at_entry", "vwap_consolidated"],
        "mae_over_atr": ["mae_over_atr", "mae_atr", "mae_x_atr"],
        "mfe_over_atr": ["mfe_over_atr", "mfe_atr", "mfe_x_atr"],
        "exit_reason": ["exit_reason", "reason", "exit_type"],
    }
    for target, aliases in col_map.items():
        for a in aliases:
            if a in df.columns:
                if target != a:
                    df[target] = df[a]
                break

    print("\n========================================================================")
    print(" OVERALL METRICS")
    print("========================================================================")
    m = _metrics(df)
    print(f"N: {m['n']}   Wins: {m['wins']}   WR: {fmt_pct(m['win_rate'])}  (95% CI {fmt_pct(m['wr_lwr'])}–{fmt_pct(m['wr_upr'])})")
    print(f"PF: {m['pf']:.3f}   Avg PnL: {m['avg_pnl']:.4f}\n")

    # ── VWAP distance grid (keep abs distance ≥ t) ────────────────────────────
    if "pct_to_vwap" in df.columns:
        print("========================================================================")
        print(" VWAP ABS DISTANCE GRID  (keep |pct_to_vwap| ≥ t)")
        print("========================================================================")
        t_grid = [0.005, 0.01, 0.015, 0.02, 0.025]  # 0.5% … 2.5%
        tbl = threshold_grid_signed(df, "pct_to_vwap", t_grid, direction=">=")
        print(tbl.to_string(index=False))
        print()

    # ── VWAP Z extreme veto (keep |z| ≤ t) ───────────────────────────────────
    if "vwap_z_at_entry" in df.columns:
        print("========================================================================")
        print(" VWAP Z EXTREME VETO GRID  (keep |z| ≤ t)")
        print("========================================================================")
        z_grid = [1.0, 1.5, 2.0, 2.5]
        tbl = threshold_grid_abs(df, "vwap_z_at_entry", z_grid, keep_op="<=")
        print(tbl.to_string(index=False))
        print()

    # ── EMA down gate (binary) ────────────────────────────────────────────────
    if "is_ema_crossed_down_at_entry" in df.columns:
        print("========================================================================")
        print(" EMA DOWN GATE  (in-gate = True)")
        print("========================================================================")
        mask = df["is_ema_crossed_down_at_entry"].astype(bool)
        tbl = binary_gate_test(df, mask, label="EMA_DOWN")
        print(tbl.to_string(index=False))
        print()

    # ── 30d return grid (keep ret_30d ≤ t) ───────────────────────────────────
    if "ret_30d_at_entry" in df.columns:
        print("========================================================================")
        print(" 30D RETURN GRID  (keep ret_30d_at_entry ≤ t)")
        print("========================================================================")
        r_grid = [0.00, 0.05, 0.10, 0.20]
        tbl = threshold_grid_signed(df, "ret_30d_at_entry", r_grid, direction="<=")
        print(tbl.to_string(index=False))
        print()

    # ── Session tests (binary) ────────────────────────────────────────────────
    if "session_tag_at_entry" in df.columns:
        print("========================================================================")
        print(" SESSION GATES  (in-gate = session)")
        print("========================================================================")
        for sess in ["ASIA", "EUROPE", "US"]:
            mask = df["session_tag_at_entry"].astype(str).str.upper().eq(sess)
            tbl = binary_gate_test(df, mask, label=f"SESSION_{sess}")
            print(tbl.to_string(index=False))
        print()

    # ── Hour pooling: block hours with WR<45% and n≥6 (example) ──────────────
    if "hour_of_day_at_entry" in df.columns:
        print("========================================================================")
        print(" HOUR POOLING  (candidate blocked hours: WR<45%, n≥6)")
        print("========================================================================")
        tmp = df.groupby("hour_of_day_at_entry")["pnl"].agg(n="count", wins=lambda s: (s>0).sum()).reset_index()
        tmp["wr"] = tmp["wins"]/tmp["n"]
        weak = tmp[(tmp["n"]>=6) & (tmp["wr"]<0.45)]["hour_of_day_at_entry"].sort_values().tolist()
        print(tmp.sort_values("hour_of_day_at_entry").to_string(index=False))
        if weak:
            print(f"\nCandidate blocked hours: {weak}")
            mask = ~df["hour_of_day_at_entry"].isin(weak)
            res = what_if_exclude(df, mask)
            print(f"What-if keep non-weak hours → N={int(mask.sum())}  WR={fmt_pct(res['win_rate'])}  PF={res['pf']:.3f}  Avg={res['avg_pnl']:.4f}")
        print()

    # ── VWAP consolidation (binary) ──────────────────────────────────────────
    if "vwap_consolidated_at_entry" in df.columns:
        print("========================================================================")
        print(" VWAP CONSOLIDATION GATE  (in-gate = consolidated=True)")
        print("========================================================================")
        mask = df["vwap_consolidated_at_entry"].astype(bool)
        tbl = binary_gate_test(df, mask, label="VWAP_CONSOLIDATED")
        print(tbl.to_string(index=False))
        print()

    # ── Payoff grid using MAE/MFE (ATR units) ────────────────────────────────
    if {"mae_over_atr", "mfe_over_atr"}.issubset(set(df.columns)):
        print("========================================================================")
        print(" PAYOFF WHAT-IF (TP/SL in ATR units; P&L in R)")
        print("========================================================================")
        tp_grid = [1.0, 1.2, 1.5, 1.8, 2.0]
        sl_grid = [1.2, 1.5, 1.8, 2.0, 2.5]
        opt, pes, mid = payoff_eval(df, tp_grid, sl_grid, tol=0.05)

        print("\n--- Optimistic (TP first when both touched) ---")
        print(opt.head(12).to_string(index=False))
        print("\n--- Pessimistic (SL first when both touched) ---")
        print(pes.head(12).to_string(index=False))
        print("\n--- Midpoint (average of optimistic & pessimistic) ---")
        print(mid.head(12).to_string(index=False))
        print("\n(Columns: reward_R = TP/SL; avg_R = mean R/trade; PF on R-units; unresolved = neither TP nor SL reached in MAE/MFE window.)")
        print()

    print("Done.")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to trades.csv")
    args = ap.parse_args()
    run(args.csv)


if __name__ == "__main__":
    main()