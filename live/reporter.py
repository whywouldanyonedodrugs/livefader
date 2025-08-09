# /opt/livefader/src/reporter.py

import asyncio
import asyncpg
import csv
import logging
import os
import argparse # Import the argument parsing library
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"

async def generate_report():
    """
    Generates either a daily or a full historical trade report based on command-line arguments.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate trade reports for the LiveFader bot.")
    parser.add_argument(
        '--full',
        action='store_true',
        help='Generate a full historical report of all closed trades.'
    )
    args = parser.parse_args()

    # --- Database Connection ---
    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found in environment or .env file. Cannot proceed.")
        return

    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)

        base_query = """
            SELECT 
                id, symbol, side, size, entry_price, stop_price, trailing_active, atr,
                status, opened_at, exit_deadline, closed_at, pnl, entry_cid, sl_cid,
                tp1_cid, tp_final_cid, sl_trail_cid, market_regime_at_entry, slippage_usd,
                risk_usd, rsi_at_entry, adx_at_entry, atr_pct_at_entry, price_boom_pct_at_entry,
                price_slowdown_pct_at_entry, vwap_dev_pct_at_entry, ret_30d_at_entry,
                ema_fast_at_entry, ema_slow_at_entry, listing_age_days_at_entry,
                session_tag_at_entry, day_of_week_at_entry, hour_of_day_at_entry,
                config_snapshot, exit_reason, holding_minutes, pnl_pct, mae_usd, mfe_usd,
                mae_over_atr, mfe_over_atr, realized_vol_during_trade, btc_beta_during_trade,
                vwap_z_at_entry, is_ema_crossed_down_at_entry, win_probability_at_entry,
                ema_spread_pct_at_entry,
                -- Counterfactual columns
                cf_would_hit_tp_2x_atr, cf_would_hit_sl_2_5x_atr,
                cf_mae_over_atr_4h, cf_mfe_over_atr_4h
            FROM positions
        """

        # --- Logic to Determine Report Type ---
        if args.full:
            report_type = "FULL HISTORICAL"
            today_utc = datetime.now(timezone.utc).date()
            filename = REPORTS_DIR / f"full_trade_history_{today_utc.isoformat()}.csv"
            
            query = """
                SELECT * FROM positions
                WHERE status = 'CLOSED'
                ORDER BY closed_at ASC
            """
            records = await conn.fetch(query)
        else:
            report_type = "DAILY"
            today_utc = datetime.now(timezone.utc).date()
            report_date = today_utc - timedelta(days=1)
            start_time = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
            end_time = start_time + timedelta(days=1)
            filename = REPORTS_DIR / f"trade_report_{report_date.isoformat()}.csv"

            query = """
                SELECT * FROM positions
                WHERE closed_at >= $1 AND closed_at < $2
                ORDER BY closed_at ASC
            """
            records = await conn.fetch(query, start_time, end_time)

        LOG.info(f"Generating {report_type} trade report...")

        if not records:
            LOG.info("No trades found for this report period. No report will be generated.")
            return

        LOG.info(f"Found {len(records)} closed trades to report.")
        REPORTS_DIR.mkdir(exist_ok=True)
        
        headers = list(records[0].keys())

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for record in records:
                writer.writerow(list(record.values()))
        
        LOG.info(f"Successfully generated report: {filename}")

    except Exception as e:
        LOG.error(f"An error occurred during report generation: {e}")
    finally:
        if conn:
            await conn.close()
            LOG.info("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(generate_report())