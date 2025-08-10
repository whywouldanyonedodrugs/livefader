# /opt/livefader/src/diagnose_db.py

import asyncio
import asyncpg
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger(__name__)

# This is the exact list of columns the backfiller tries to update.
# We will compare this list to what the database reports.
EXPECTED_COLUMNS = [
    'pnl', 'pnl_pct', 'exit_reason', 'holding_minutes', 'rsi_at_entry',
    'adx_at_entry', 'ema_fast_at_entry', 'ema_slow_at_entry', 'ret_30d_at_entry',
    'mae_usd', 'mfe_usd', 'mae_over_atr', 'mfe_over_atr', 'vwap_dev_pct_at_entry',
    'vwap_z_at_entry', 'is_ema_crossed_down_at_entry', 'ema_spread_pct_at_entry',
    'vwap_consolidated_at_entry', 'eth_macd_at_entry', 'eth_macdsignal_at_entry',
    'eth_macdhist_at_entry', 'eth_macd_1h_at_entry', 'eth_macdsignal_1h_at_entry',
    'eth_macdhist_1h_at_entry'
]

async def main():
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found in .env file.")
        return

    LOG.info(f"Attempting to connect using DSN from .env file: {db_dsn}")
    
    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        
        LOG.info("Connection successful. Running diagnostics...")
        
        current_db = await conn.fetchval("SELECT current_database();")
        LOG.info(f"SCRIPT IS CONNECTED TO DATABASE: '{current_db}'")
        
        LOG.info("Fetching schema for 'positions' table as seen by this script...")
        
        schema_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'positions'
            ORDER BY ordinal_position;
        """
        schema_records = await conn.fetch(schema_query)
        
        actual_columns = {record['column_name'] for record in schema_records}
        
        print("\n--- SCHEMA AS SEEN BY SCRIPT ---")
        for col in sorted(list(actual_columns)):
            print(f"- {col}")
        print("--- END OF SCHEMA ---\n")

        # --- THE FINAL VERDICT ---
        LOG.info("Comparing expected columns from the script with actual columns from the database...")
        
        expected_set = set(EXPECTED_COLUMNS)
        
        missing_columns = expected_set - actual_columns
        
        if not missing_columns:
            LOG.info("SUCCESS: All expected columns were found in the database schema.")
            LOG.info("This indicates a very unusual caching issue with the asyncpg driver's prepared statements.")
        else:
            LOG.error("CRITICAL FAILURE: The database this script is connecting to is MISSING the following columns:")
            for col in sorted(list(missing_columns)):
                print(f"  - {col}")
            LOG.error("This confirms an environment or DSN mismatch.")

    except Exception as e:
        LOG.error(f"A critical error occurred during diagnostics: {e}", exc_info=True)
    finally:
        if conn: await conn.close()
        LOG.info("Diagnostic script finished.")

if __name__ == "__main__":
    asyncio.run(main())