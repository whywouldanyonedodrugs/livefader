# live/database.py
import asyncpg
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

# --- MODIFIED: Added all new columns to the table creation script ---
TABLES_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    size NUMERIC,
    entry_price NUMERIC,
    stop_price NUMERIC,
    trailing_active BOOLEAN DEFAULT FALSE,
    atr NUMERIC,
    status TEXT,
    opened_at TIMESTAMPTZ,
    exit_deadline TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    pnl NUMERIC,
    -- New columns for data analysis
    entry_cid VARCHAR(36),
    sl_cid VARCHAR(36),
    tp1_cid VARCHAR(36),
    tp_final_cid VARCHAR(36),
    sl_trail_cid VARCHAR(36),
    market_regime_at_entry TEXT,
    slippage_usd FLOAT,
    risk_usd FLOAT,
    rsi_at_entry FLOAT,
    adx_at_entry FLOAT,
    atr_pct_at_entry FLOAT,
    price_boom_pct_at_entry FLOAT,
    price_slowdown_pct_at_entry FLOAT,
    vwap_dev_pct_at_entry FLOAT,
    ret_30d_at_entry FLOAT,
    ema_fast_at_entry FLOAT,
    ema_slow_at_entry FLOAT,
    listing_age_days_at_entry INT,
    session_tag_at_entry TEXT,
    day_of_week_at_entry INT,
    hour_of_day_at_entry INT,
    config_snapshot JSONB,
    exit_reason TEXT,
    holding_minutes FLOAT,
    pnl_pct FLOAT,
    mae_usd FLOAT,
    mfe_usd FLOAT,
    mae_over_atr FLOAT,
    mfe_over_atr FLOAT,
    realized_vol_during_trade FLOAT,
    btc_beta_during_trade FLOAT, 
    -- Columns for the predictive model
    vwap_z_at_entry FLOAT,
    is_ema_crossed_down_at_entry BOOLEAN,
    ema_spread_pct_at_entry FLOAT,
    win_probability_at_entry FLOAT,
    cf_would_hit_tp_1x_atr BOOLEAN,
    cf_would_hit_tp_2x_atr BOOLEAN,
    cf_would_hit_sl_2_5x_atr BOOLEAN,
    cf_mae_over_atr_4h FLOAT,
    cf_mfe_over_atr_4h FLOAT
);
CREATE TABLE IF NOT EXISTS fills (
    id SERIAL PRIMARY KEY,
    position_id INT REFERENCES positions(id),
    fill_type TEXT,
    price NUMERIC,
    qty NUMERIC,
    ts TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TIMESTAMPTZ PRIMARY KEY,
    equity NUMERIC
);

-- New indexes for performance
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions (symbol, status);
CREATE INDEX IF NOT EXISTS idx_positions_closed_at ON positions (closed_at);
"""
DB_RETRYABLE_EXCEPTIONS = (
    asyncpg.exceptions.InterfaceError,
    asyncpg.exceptions.DeadlockDetectedError,
    asyncpg.exceptions.SerializationError,
)

db_retry = retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(DB_RETRYABLE_EXCEPTIONS),
    before_sleep=lambda state: LOG.warning(
        "Retrying DB call %s due to %s. Attempt #%d",
        state.fn.__name__, state.outcome.exception(), state.attempt_number
    )
)

class DB:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    @db_retry
    async def init(self):
        self.pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self.pool.acquire() as conn:
            await conn.execute(TABLES_SQL)

    @db_retry
    async def insert_position(self, data: Dict[str, Any]) -> int:
        # This query is now dynamically built to handle all fields present in the data dict
        columns = data.keys()
        values_placeholder = ", ".join([f"${i+1}" for i in range(len(columns))])
        query = f"""
            INSERT INTO positions ({", ".join(columns)})
            VALUES ({values_placeholder})
            RETURNING id
        """
        return await self.pool.fetchval(query, *data.values())

    @db_retry
    async def update_position(self, pid: int, **fields):
        sets = ",".join(f"{k}=${i+2}" for i, k in enumerate(fields))
        await self.pool.execute(
            f"UPDATE positions SET {sets} WHERE id=$1", pid, *fields.values()
        )

    # ... (The rest of your database.py file is correct and does not need changes) ...

    # --- UNCHANGED: The rest of the file is correct ---
    @db_retry
    async def add_fill(self, pid: int, fill_type: str, price: Optional[float], qty: float, ts: datetime):
        await self.pool.execute(
            "INSERT INTO fills(position_id,fill_type,price,qty,ts) VALUES($1,$2,$3,$4,$5)",
            pid, fill_type, price, qty, ts
        )

    @db_retry
    async def fetch_open_positions(self) -> List[asyncpg.Record]:
        return await self.pool.fetch("SELECT * FROM positions WHERE status='OPEN'")

    @db_retry
    async def latest_equity(self) -> Optional[float]:
        return await self.pool.fetchval(
            "SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 1"
        )

    @db_retry
    async def snapshot_equity(self, equity: float, ts: datetime):
        await self.pool.execute(
            "INSERT INTO equity_snapshots VALUES($1,$2) ON CONFLICT DO NOTHING",
            ts, equity
        )

    @db_retry
    async def batch_insert_fills(self, fills_data: list[tuple]):
        if not fills_data:
            return
        q = "INSERT INTO fills(position_id, fill_type, price, qty, ts) VALUES($1, $2, $3, $4, $5)"
        try:
            await self.pool.executemany(q, fills_data)
            LOG.info("Successfully batch-inserted %d fill records.", len(fills_data))
        except Exception as e:
            LOG.error("Failed to batch-insert fills: %s", e)