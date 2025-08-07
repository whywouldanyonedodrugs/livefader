# /opt/livefader/src/analyze_trades.py

import argparse
from pathlib import Path
import warnings
import os  # <--- THE PRIMARY FIX IS HERE
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import telegram
from dotenv import load_dotenv

try:
    import statsmodels.api as sm
except ImportError:
    sm = None
    warnings.warn("statsmodels not installed – logistic regression section will be skipped.")

# --- Helper Functions ---
def header(txt: str):
    bar = "=" * 72
    print(f"\n{bar}\n {txt.upper()}\n{bar}")

def describe_continuous(df, col, win_flag="is_win"):
    header(f"{col} – wins vs. losses")
    grp = df.groupby(win_flag)[col].describe()[["count", "mean", "std", "25%", "50%", "75%"]]
    print(grp.to_string())
    if df[col].notna().sum() > 10:
        try:
            r, p = stats.pointbiserialr(df[win_flag].fillna(False), df[col].fillna(0))
            print(f"\nPoint-biserial correlation with win flag: r={r:.3f}  p={p:.4f}")
        except ValueError:
            print("\nCould not calculate point-biserial correlation.")

def describe_categorical(df, col, win_flag="is_win"):
    header(f"{col} – win rate by category")
    tab = pd.crosstab(df[col], df[win_flag], margins=True)
    if True in tab.columns:
        tab["win_rate_%"] = 100 * tab[True] / tab["All"]
    print(tab.to_string())
    if len(tab) > 1 and tab.shape[0] > 2 and tab.shape[1] > 2:
        chi2 = stats.chi2_contingency(tab.iloc[:-1, :-1])[0]
        n = tab["All"].iloc[-1]
        k = min(tab.shape[0] - 1, tab.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * k)) if n and k else np.nan
        print(f"\nCramér’s V vs. win flag: {cramers_v:.3f}")

def logistic_regression(df, features):
    if sm is None: return
    header("Quick logistic regression (wins=1)")
    clean_df = df.dropna(subset=features + ["is_win"])
    if len(clean_df) < 10:
        print("Not enough data for logistic regression.")
        return
    X = clean_df[features]
    X = sm.add_constant(X)
    y = clean_df["is_win"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())
    except Exception as e:
        print(f"Logistic regression failed: {e}")

def analyze_regime_performance(df: pd.DataFrame):
    header("Regime-Specific Performance KPIs")
    if "market_regime_at_entry" not in df.columns: return
    regime_groups = df.groupby("market_regime_at_entry")
    summary = regime_groups.agg(
        total_trades=('pnl', 'count'), total_pnl=('pnl', 'sum'),
        win_rate=('is_win', lambda x: x.mean() * 100), avg_pnl=('pnl', 'mean'),
        avg_win=('pnl', lambda x: x[x > 0].mean()), avg_loss=('pnl', lambda x: x[x < 0].mean()),
    ).fillna(0)
    gross_profit = regime_groups['pnl'].apply(lambda x: x[x > 0].sum())
    gross_loss = regime_groups['pnl'].apply(lambda x: abs(x[x < 0].sum()))
    summary['profit_factor'] = (gross_profit / gross_loss).replace([np.inf, -np.inf], 0).fillna(0)
    print(summary.to_string(float_format="%.2f"))

# --- Core Analysis Logic ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_win"] = df["pnl"] > 0
    if "slippage_usd" in df.columns and "atr" in df.columns:
        df["slippage_abs"] = df["slippage_usd"].abs()
        df["slippage_over_atr"] = df["slippage_abs"].divide(df["atr"]).replace([np.inf, -np.inf], np.nan)
    else:
        df["slippage_over_atr"] = np.nan
    df["price_minus_ema_fast"] = df["entry_price"] - df["ema_fast_at_entry"]
    df["price_minus_ema_slow"] = df["entry_price"] - df["ema_slow_at_entry"]
    df["pct_to_ema_fast"] = df["price_minus_ema_fast"] / df["entry_price"]
    df["pct_to_ema_slow"] = df["price_minus_ema_slow"] / df["entry_price"]
    df["ema_spread_abs"] = df["ema_fast_at_entry"] - df["ema_slow_at_entry"]
    df["ema_spread_pct"] = df["ema_spread_abs"] / df["ema_slow_at_entry"]
    df.rename(columns={"vwap_dev_pct_at_entry": "pct_to_vwap"}, inplace=True)
    df["holding_hours"] = df["holding_minutes"] / 60.0
    df["holding_bucket"] = pd.cut(
        df["holding_hours"],
        bins=[0, 1, 2, 4, 8, 24, np.inf],
        labels=["<1 h", "1-2 h", "2-4 h", "4-8 h", "8-24 h", ">24 h"],
    )
    return df

def run_analysis(df: pd.DataFrame):
    df = feature_engineering(df)
    
    header("Price boom / slowdown bins")
    df["boom_bin"] = pd.qcut(df["price_boom_pct_at_entry"], q=4, labels=False, duplicates="drop")
    df["slow_bin"] = pd.qcut(df["price_slowdown_pct_at_entry"], q=4, labels=False, duplicates="drop")
    print(pd.crosstab(df["boom_bin"], df["slow_bin"], values=df["is_win"], aggfunc="mean"))

    analyze_regime_performance(df)

    df["rsi_band"] = pd.cut(df["rsi_at_entry"], bins=[0, 50, 60, 70, 80, 100], labels=["<50", "50-60", "60-70", "70-80", ">80"])
    describe_categorical(df, "rsi_band")
    
    describe_continuous(df, "pct_to_ema_fast")
    describe_continuous(df, "ema_spread_pct")
    describe_continuous(df, "pct_to_vwap")
    describe_continuous(df, "mae_over_atr")
    describe_continuous(df, "mfe_over_atr")
    describe_continuous(df, "realized_vol_during_trade")
    describe_continuous(df, "btc_beta_during_trade")
    describe_continuous(df, "slippage_over_atr")
    
    describe_categorical(df, "session_tag_at_entry")
    describe_categorical(df, "holding_bucket")
    describe_categorical(df, "day_of_week_at_entry")
    describe_categorical(df, "hour_of_day_at_entry")

    header("Top 10 absolute point-biserial correlations with win flag")
    cont_cols = [c for c in df.select_dtypes(include="number").columns if c not in ["is_win", "pnl"]]
    cors = {c: stats.pointbiserialr(df["is_win"], df[c])[0] for c in cont_cols if df[c].notna().sum() > 5 and np.std(df[c]) > 0}
    top10 = pd.Series(cors).abs().sort_values(ascending=False).head(10)
    print(top10.to_string())

    logit_features = [col for col in top10.index.tolist() if col not in ['pnl_pct', 'mae_usd', 'mfe_usd']]
    logistic_regression(df, logit_features)

def analyze_performance_ratios(df_equity: pd.DataFrame):
    header("Risk-Adjusted Performance Ratios (Sharpe & Sortino)")
    if df_equity.empty or len(df_equity) < 2:
        print("Not enough equity data to calculate performance ratios.")
        return
    df_equity['ts'] = pd.to_datetime(df_equity['ts'])
    df_equity = df_equity.sort_values('ts').set_index('ts')
    daily_equity = df_equity['equity'].resample('D').last()
    daily_returns = daily_equity.pct_change().dropna()
    if len(daily_returns) < 2:
        print("Not enough daily data points to calculate ratios.")
        return
    mean_daily_return = daily_returns.mean()
    std_dev_return = daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    annual_return = mean_daily_return * 365
    annual_std_dev = std_dev_return * np.sqrt(365)
    annual_downside_std = downside_std * np.sqrt(365)
    sharpe_ratio = annual_return / annual_std_dev if annual_std_dev != 0 else 0
    sortino_ratio = annual_return / annual_downside_std if annual_downside_std != 0 else 0
    print(f"Total Days Analyzed:      {len(daily_returns)}")
    print(f"Annualized Return:        {annual_return:.2%}")
    print(f"Annualized Volatility:    {annual_std_dev:.2%}")
    print(f"Annualized Downside Vol.: {annual_downside_std:.2%}")
    print("-" * 40)
    print(f"Sharpe Ratio:             {sharpe_ratio:.2f}")
    print(f"Sortino Ratio:            {sortino_ratio:.2f}")

# --- Telegram Delivery Function ---
async def send_telegram_report(report_file_path: str):
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
                filename="weekly_report.txt",
                caption=f"LiveFader Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"
            )
        print("Telegram report sent successfully.")
    except FileNotFoundError:
        print(f"Report file not found at {report_file_path}.")
    except Exception as e:
        print(f"Failed to send report to Telegram: {e}")

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Trade analytics console report")
    parser.add_argument("--file", default="trades.csv", help="CSV of closed trades")
    parser.add_argument("--equity-file", default="equity_data.csv", help="CSV of equity snapshots")
    parser.add_argument("--send-report", help="Path to the report file to be sent via Telegram")
    args = parser.parse_args()

    if args.send_report:
        asyncio.run(send_telegram_report(args.send_report))
        return

    path = Path(args.file)
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            run_analysis(df)
    else:
        print(f"Trade file not found: {path}")

    equity_path = Path(args.equity_file)
    if equity_path.exists():
        df_equity = pd.read_csv(equity_path)
        analyze_performance_ratios(df_equity)
    else:
        print(f"\nEquity file not found: {equity_path}. Skipping performance ratio analysis.")

if __name__ == "__main__":
    main()