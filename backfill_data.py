# /opt/livefader/src/backfill_data.py

import asyncio
import asyncpg
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import ccxt.async_support as ccxt
from tqdm import tqdm

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

# --- Replicated Logic from your bot ---
# We need these functions here so the script can perform the same calculations.

def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean().rolling(window=period, min_periods=period).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = pd.DataFrame(df['high'] - df['low'])
    tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
    tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

async def calculate_historical_regime(exchange, opened_at: datetime) -> str:
    """Calculates the market regime for a specific historical date."""
    try:
        since_ts = int((opened_at - timedelta(days=500)).timestamp() * 1000)
        ohlcv = await exchange.fetch_ohlcv('BTCUSDT', '1d', since=since_ts, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[df['timestamp'] < opened_at] # Use data available *before* the trade
        df.set_index('timestamp', inplace=True)

        daily_returns = df['close'].pct_change().dropna()
        
        # Volatility Regime
        model = sm.tsa.MarkovRegression(daily_returns, k_regimes=2, switching_variance=True)
        results = model.fit(disp=False)
        low_vol_regime_idx = np.argmin(results.params[-2:])
        vol_regime = "LOW_VOL" if results.smoothed_marginal_probabilities[low_vol_regime_idx].iloc[-1] > 0.5 else "HIGH_VOL"

        # Trend Regime
        df['tma'] = triangular_moving_average(df['close'], 100)
        atr_series = atr(df, period=20)
        df['keltner_upper'] = df['tma'] + (atr_series * 2.0)
        df['keltner_lower'] = df['tma'] - (atr_series * 2.0)
        df.dropna(inplace=True)
        
        last_day = df.iloc[-1]
        if last_day['close'] > last_day['keltner_upper']:
            trend_regime = "BULL"
        elif last_day['close'] < last_day['keltner_lower']:
            trend_regime = "BEAR"
        else:
            trend_regime = "UNKNOWN" # Can't easily determine trend in the middle

        return f"{trend_regime}_{vol_regime}"
    except Exception as e:
        LOG.warning(f"Could not calculate historical regime for {opened_at.date()}: {e}")
        return "UNKNOWN"

async def main():
    """
    Finds trades with missing data and backfills them by fetching historical data.
    """
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found. Cannot proceed.")
        return

    conn = None
    exchange = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        exchange = ccxt.bybit({'enableRateLimit': True})
        
        # Find all trades where the new data is missing
        query = "SELECT * FROM positions WHERE market_regime_at_entry IS NULL AND status = 'CLOSED' ORDER BY id"
        trades_to_fix = await conn.fetch(query)

        if not trades_to_fix:
            LOG.info("No trades found with missing data. Database is up-to-date.")
            return

        LOG.info(f"Found {len(trades_to_fix)} trades to backfill. Starting process...")

        for trade in tqdm(trades_to_fix, desc="Backfilling Trades"):
            try:
                trade_id = trade['id']
                symbol = trade['symbol']
                opened_at = trade['opened_at']
                closed_at = trade['closed_at']
                entry_price = float(trade['entry_price'])
                size = float(trade['size'])

                # --- 1. Calculate Pre-Trade Metrics ---
                since_ts = int((opened_at - timedelta(days=35)).timestamp() * 1000)
                ohlcv_5m = await exchange.fetch_ohlcv(symbol, '5m', since=since_ts, limit=1000)
                df5 = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df5['timestamp'] = pd.to_datetime(df5['timestamp'], unit='ms', utc=True)
                df5 = df5[df5['timestamp'] <= opened_at].tail(288) # Get last day of 5m data

                # Note: This is a simplified recreation. Some values might be slightly different.
                atr_series = atr(df5, period=14)
                atr_at_entry = atr_series.iloc[-1]
                atr_pct_at_entry = (atr_at_entry / entry_price) * 100 if entry_price > 0 else 0
                
                # --- 2. Calculate Market Regime ---
                market_regime = await calculate_historical_regime(exchange, opened_at)

                # --- 3. Calculate Post-Trade Metrics ---
                holding_minutes = (closed_at - opened_at).total_seconds() / 60
                
                # Fetch 1-min data for MAE/MFE
                ohlcv_1m = await exchange.fetch_ohlcv(symbol, '1m', since=int(opened_at.timestamp() * 1000))
                df1 = pd.DataFrame(ohlcv_1m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df1['ts'] = pd.to_datetime(df1['ts'], unit='ms', utc=True)
                df1 = df1[(df1['ts'] >= opened_at) & (df1['ts'] <= closed_at)]
                
                mae_usd, mfe_usd = 0.0, 0.0
                if not df1.empty:
                    high_during_trade = df1['h'].max()
                    low_during_trade = df1['l'].min()
                    mae_usd = (high_during_trade - entry_price) * size
                    mfe_usd = (entry_price - low_during_trade) * size

                # --- 4. Update the Database Record ---
                update_query = """
                    UPDATE positions SET
                        market_regime_at_entry = $1,
                        atr_pct_at_entry = $2,
                        holding_minutes = $3,
                        mae_usd = $4,
                        mfe_usd = $5
                    WHERE id = $6
                """
                await conn.execute(update_query, market_regime, atr_pct_at_entry, holding_minutes, mae_usd, mfe_usd, trade_id)

            except Exception as e:
                LOG.error(f"Failed to backfill trade ID {trade['id']} ({trade['symbol']}): {e}")
                continue

        LOG.info("Backfilling process complete.")

    except Exception as e:
        LOG.error(f"A critical error occurred during the backfill process: {e}")
    finally:
        if conn:
            await conn.close()
        if exchange:
            await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())