# regime_detector.py
"""
Calculates and saves the daily market regime based on a benchmark asset.

Methodology inspired by Srivastava & Bhattacharyya (2018):
1.  Volatility Regime: A two-state Markov Switching model is fitted to daily
    returns to identify 'High Vol' and 'Low Vol' periods.
2.  Trend Regime: A Keltner Channel around a long-term Triangular Moving Average (TMA)
    is used to identify 'Bull' and 'Bear' trends, avoiding whipsaws.
3.  Combined Regime: The two are combined to create four final regimes:
    - BULL_LOW_VOL
    - BULL_HIGH_VOL
    - BEAR_LOW_VOL
    - BEAR_HIGH_VOL
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

import config as cfg
import shared_utils
import indicators as ta

def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Calculates a Triangular Moving Average (TMA)."""
    return series.rolling(window=period).mean().rolling(window=period).mean()

def main():
    """Runs the regime detection process and saves the output."""
    print(f"Starting regime detection using benchmark: {cfg.REGIME_BENCHMARK_SYMBOL}")

    # --- 1. Load and Prepare Benchmark Data ---
    try:
        df = shared_utils.load_parquet_data(cfg.REGIME_BENCHMARK_SYMBOL)
    except FileNotFoundError:
        print(f"ERROR: Benchmark symbol '{cfg.REGIME_BENCHMARK_SYMBOL}.parquet' not found.")
        print("Please ensure you have data for the benchmark and have run etl.py.")
        return

    df_daily = df.resample('D').agg({'close': 'last'}).dropna()
    daily_returns = df_daily['close'].pct_change().dropna()

    # --- 2. Volatility Regime (Markov Switching Model) ---
    print("Fitting Markov Switching model for volatility regime...")
    model = sm.tsa.MarkovRegression(
        daily_returns, k_regimes=2, switching_variance=True, trend='c'
    )
    
    # Initialize the column before trying to fill it
    df_daily['vol_regime'] = "UNKNOWN"
    try:
        results = model.fit(disp='iter')
        low_vol_regime = np.argmin(results.params[-2:])
        
        # Calculate the regimes for the returns data
        vol_regimes_for_returns = np.where(results.smoothed_marginal_probabilities[low_vol_regime] > 0.5, "LOW_VOL", "HIGH_VOL")
        
        # --- THIS IS THE CORRECTED ASSIGNMENT ---
        # Assign the results back to the main df using the index from the returns series
        df_daily.loc[daily_returns.index, 'vol_regime'] = vol_regimes_for_returns
        print("Volatility regime detection complete.")
    except Exception as e:
        print(f"Could not fit Markov model: {e}. Proceeding without volatility regime.")
        # If it fails, the column remains "UNKNOWN"

    # --- 3. Trend Regime (TMA + Keltner Channel) ---
    print("Calculating trend regime using TMA and Keltner Channel...")
    df_daily['tma'] = triangular_moving_average(df_daily['close'], cfg.REGIME_MA_PERIOD)
    daily_ohlc = df.resample('D').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    atr_series = ta.atr(daily_ohlc, period=cfg.REGIME_ATR_PERIOD)
    df_daily['atr'] = atr_series.reindex(df_daily.index, method='ffill')
    df_daily['keltner_upper'] = df_daily['tma'] + (df_daily['atr'] * cfg.REGIME_ATR_MULT)
    df_daily['keltner_lower'] = df_daily['tma'] - (df_daily['atr'] * cfg.REGIME_ATR_MULT)
    df_daily.dropna(inplace=True)

    # --- FIX FOR FUTUREWARNING ---
    # Initialize the Series with dtype='object' to handle strings without warnings
    trend = pd.Series(np.nan, index=df_daily.index, dtype="object")
    
    for i in tqdm(range(1, len(df_daily)), desc="Confirming Trend"):
        if df_daily['close'].iloc[i] > df_daily['keltner_upper'].iloc[i]:
            trend.iloc[i] = "BULL"
        elif df_daily['close'].iloc[i] < df_daily['keltner_lower'].iloc[i]:
            trend.iloc[i] = "BEAR"
        else:
            trend.iloc[i] = trend.iloc[i-1]

    df_daily['trend_regime'] = trend.ffill().bfill()
    print("Trend regime detection complete.")

    # --- 4. Combine and Save ---
    df_daily['regime'] = df_daily['trend_regime'] + "_" + df_daily['vol_regime']
    
    output_path = cfg.PROJECT_ROOT / "regime_data.parquet"
    final_df = df_daily[['regime']]
    final_df.to_parquet(output_path)

    print(f"\nRegime detection finished. Data saved to {output_path}")
    print("Regime counts:")
    print(final_df['regime'].value_counts())

if __name__ == "__main__":
    main()