# /opt/livefader/src/reporter.py

import asyncio
import asyncpg
import csv
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# --- Basic Configuration ---
# Set up logging to see the script's output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

# Define the output directory for the reports
REPORTS_DIR = Path(__file__).parent / "reports"

async def generate_daily_csv_report():
    """
    Connects to the database, fetches all trades closed in the previous UTC day,
    and writes them to a detailed CSV file.
    """
    load_dotenv() # Load environment variables from .env file
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found in environment or .env file. Cannot proceed.")
        return

    # Define the time range for the report (the entire previous day in UTC)
    today_utc = datetime.now(timezone.utc).date()
    report_date = today_utc - timedelta(days=1)
    start_time = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=1)

    LOG.info(f"Generating daily trade report for trades closed on: {report_date.isoformat()}")

    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        
        # This query selects all relevant columns for trades closed within the time range
        query = """
            SELECT
                id,
                symbol,
                side,
                status,
                opened_at,
                closed_at,
                pnl,
                entry_price,
                size,
                market_regime_at_entry,
                slippage_usd,
                rsi_at_entry,
                atr_pct_at_entry,
                price_boom_pct_at_entry,
                price_slowdown_pct_at_entry,
                exit_reason,
                holding_minutes
            FROM positions
            WHERE closed_at >= $1 AND closed_at < $2
            ORDER BY closed_at ASC
        """
        
        records = await conn.fetch(query, start_time, end_time)

        if not records:
            LOG.info("No trades were closed on this date. No report will be generated.")
            return

        LOG.info(f"Found {len(records)} closed trades to report.")

        # Create the reports directory if it doesn't exist
        REPORTS_DIR.mkdir(exist_ok=True)
        
        # Define the CSV filename
        filename = REPORTS_DIR / f"trade_report_{report_date.isoformat()}.csv"
        
        # Get headers dynamically from the first record
        headers = list(records[0].keys())

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(headers)
            # Write the data rows
            for record in records:
                writer.writerow(list(record.values()))
        
        LOG.info(f"Successfully generated daily CSV report: {filename}")

    except Exception as e:
        LOG.error(f"An error occurred during report generation: {e}")
    finally:
        if conn:
            await conn.close()
            LOG.info("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(generate_daily_csv_report())