# /opt/livefader/src/train_model.py

import asyncio
import asyncpg
import logging
import os
from dotenv import load_dotenv
import pandas as pd
import statsmodels.api as sm
import joblib # For saving the model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

# --- CRITICAL: Final, Corrected Feature Selection ---
# These are the pre-trade features the bot will use to predict win probability.
# The names must exactly match the column names in the database.
FEATURES = [
    # Core Indicators
    'rsi_at_entry',
    'adx_at_entry',
    
    # Price Action / Momentum
    'price_boom_pct_at_entry',
    'price_slowdown_pct_at_entry',
    
    # Volatility-Normalized Mean Reversion
    'vwap_z_at_entry',
    
    # Trend Context
    'ema_spread_pct_at_entry', # <-- CORRECTED: Added '_at_entry' suffix
    'is_ema_crossed_down_at_entry',
    
    # Time-Based Features
    'day_of_week_at_entry',
    'hour_of_day_at_entry'
]
TARGET = 'is_win'
MODEL_FILE_NAME = "win_probability_model.joblib"


async def main():
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found. Cannot proceed.")
        return

    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        
        LOG.info("Fetching clean trade history from the database...")
        query = "SELECT * FROM positions WHERE status = 'CLOSED'"
        data = await conn.fetch(query)
        
        if not data:
            LOG.error("No trade data found in the database. Cannot train model.")
            return

        df = pd.DataFrame([dict(record) for record in data])
            
        df['is_win'] = df['pnl'] > 0
        LOG.info(f"Loaded {len(df)} trades for training.")

        # --- THE DEFINITIVE FIX ---
        # 1. Force all feature columns to a numeric type as a safety measure.
        for col in FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                LOG.error(f"Critical error: Feature column '{col}' not found in the DataFrame.")
                return
        
        # 2. Explicitly convert the boolean feature to integers (1/0).
        # This is the key step that solves the 'dtype of object' error.
        if 'is_ema_crossed_down_at_entry' in df.columns:
            df['is_ema_crossed_down_at_entry'] = df['is_ema_crossed_down_at_entry'].astype(int)
        # --- END OF FIX ---

        # --- Model Training ---
        # Prepare the data, dropping any rows with missing values
        df_clean = df.dropna(subset=FEATURES + [TARGET])
        
        X = df_clean[FEATURES]
        y = df_clean[TARGET].astype(int) # Also cast the target to int for consistency

        # Add a constant for the regression model
        X = sm.add_constant(X, prepend=True)

        LOG.info(f"Training Firth logistic regression model on {len(X)} data points...")
        
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(method='firth', disp=False)

        LOG.info("Model training complete.")
        print("--- Firth Logit Regression Results ---")
        print(result.summary())

        # --- Save the Trained Model ---
        joblib.dump(result, MODEL_FILE_NAME)
        LOG.info(f"Successfully saved trained model to '{MODEL_FILE_NAME}'")

    except Exception as e:
        LOG.error(f"An error occurred during model training: {e}", exc_info=True)
    finally:
        if conn: await conn.close()

if __name__ == "__main__":
    asyncio.run(main())