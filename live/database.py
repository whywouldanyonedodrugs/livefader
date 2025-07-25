# live/database.py
import asyncpg
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

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
    pnl NUMERIC
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
    asyncpg.exceptions.InterfaceError,       # Connection errors
    asyncpg.exceptions.DeadlockDetectedError,  # Deadlocks
    asyncpg.exceptions.SerializationError,     # Transaction conflicts
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
        q = """INSERT INTO positions(symbol,side,size,entry_price,stop_price,
                 trailing_active,atr,status,opened_at, exit_deadline)
               VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING id"""
        return await self.pool.fetchval(
            q,
            data["symbol"], data["side"], data["size"],
            data["entry_price"], data["stop_price"], data["trailing_active"],
            data["atr"], data["status"], data["opened_at"], data["exit_deadline"]
        )

    @db_retry
    async def update_position(self, pid: int, **fields):
        sets = ",".join(f"{k}=${i+2}" for i, k in enumerate(fields))
        await self.pool.execute(
            f"UPDATE positions SET {sets} WHERE id=$1", pid, *fields.values()
        )

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