# research/data/tca.py
import numpy as np
import pandas as pd

def approx_fee_paid_usd(row: pd.Series, taker_bps: float = 5.0) -> float:
    """
    Rough taker fee estimate in USD: notional * fee_rate.
    Adjust taker_bps to your account rate (default 5 bps = 0.05%).
    """
    notional = float(abs(row.get("size", 0.0)) * float(row.get("entry_price", 0.0)))
    return notional * (taker_bps / 1e4)

def realized_funding_usd(row: pd.Series, funding_rate: float, hours_held: float) -> float:
    """
    Approx funding transfer for perps:
      funding_pnl ≈ notional * funding_rate * (hours_held / funding_interval_hours)
    Most perps fund every 8h (varies by symbol; we use 8h heuristic).
    """
    interval_h = 8.0
    notional = float(abs(row.get("size", 0.0)) * float(row.get("entry_price", 0.0)))
    return notional * float(funding_rate) * (hours_held / interval_h)

def add_cost_audit(df: pd.DataFrame, funding_col: str = "funding_last_at_entry", taker_bps: float = 5.0) -> pd.DataFrame:
    """
    Add rough fee + funding audit columns so reports can decompose P&L.
    (We use funding at entry as proxy; refine later with actual funding legs if stored.)
    """
    out = df.copy()
    open_ts = pd.to_datetime(out["opened_at"], utc=True)
    close_ts = pd.to_datetime(out["closed_at"], utc=True)
    hours = (close_ts - open_ts).dt.total_seconds() / 3600.0

    out["fee_usd_est"] = out.apply(lambda r: approx_fee_paid_usd(r, taker_bps=taker_bps), axis=1)
    out["funding_usd_est"] = [
        realized_funding_usd(r, r.get(funding_col, 0.0) or 0.0, h) for r, h in zip(out.to_dict("records"), hours)
    ]
    return out
