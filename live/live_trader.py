"""
live_trader.py – v1.2.1
===========================================================================

"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import signal as sigmod
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncpg
import ccxt.async_support as ccxt
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config as cfg   # <‑‑ your static defaults
import filters
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .signal_generator import SignalGenerator, Signal
from .exchange_proxy import ExchangeProxy

from pydantic import BaseSettings, Field, ValidationError

# ---------------------------  YOUR MODULES  ------------------------------
import backtest.scout as scout  # async def scan_symbol(sym: str, cfg: dict) -> Optional[Signal]
# -------------------------------------------------------------------------

LISTING_PATH = Path("listing_dates.json")

###############################################################################
# 0 ▸ LOGGING #################################################################
###############################################################################

LOG = logging.getLogger("live_trader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

###############################################################################
# 1 ▸ SETTINGS pulled from ENV (.env) #########################################
###############################################################################


class Settings(BaseSettings):
    """Secrets & env flags."""

    bybit_api_key: str = Field(..., env="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., env="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(False, env="BYBIT_TESTNET")

    tg_bot_token: str = Field(..., env="TG_BOT_TOKEN")
    tg_chat_id: str = Field(..., env="TG_CHAT_ID")

    pg_dsn: str = Field(..., env="DATABASE_URL")

    default_leverage: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False


###############################################################################
# 2 ▸ PATHS & YAML LOADER #####################################################
###############################################################################

CONFIG_PATH = Path("config.yaml")
SYMBOLS_PATH = Path("symbols.txt")


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text()) or {}


###############################################################################
# 3 ▸ DB LAYER ################################################################
###############################################################################

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

# Create a retry decorator for database operations
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

    # ---------- helper wrappers ------------------------------------------
    @db_retry
    async def insert_position(self, data: Dict[str, Any]) -> int:
        q = """INSERT INTO positions(symbol,side,size,entry_price,stop_price,
                 trailing_active,atr,status,opened_at)
               VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id"""
        return await self.pool.fetchval(
            q,
            data["symbol"],
            data["side"],
            data["size"],
            data["entry_price"],
            data["stop_price"],
            data["trailing_active"],
            data["atr"],
            data["status"],
            data["opened_at"],
        )

    @db_retry
    async def update_position(self, pid: int, **fields):
        sets = ",".join(f"{k}=${i+2}" for i, k in enumerate(fields))
        await self.pool.execute(
            f"UPDATE positions SET {sets} WHERE id=$1", pid, *fields.values()
        )

    @db_retry
    async def add_fill(self, pid: int, fill_type: str, price: Optional[float], qty: float):
        await self.pool.execute(
            "INSERT INTO fills(position_id,fill_type,price,qty,ts) VALUES($1,$2,$3,$4,$5)",
            pid,
            fill_type,
            price,
            qty,
            datetime.now(timezone.utc),
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
    async def snapshot_equity(self, equity: float):
        await self.pool.execute(
            "INSERT INTO equity_snapshots VALUES($1,$2) ON CONFLICT DO NOTHING",
            datetime.now(timezone.utc),
            equity,
        )

    @db_retry
    async def batch_insert_fills(self, fills_data: list[tuple]):
        """
        Inserts multiple fill records in a single database transaction.
        This is much more efficient than inserting one by one.
        
        Args:
            fills_data: A list of tuples, where each tuple matches the fills table:
                        (position_id, fill_type, price, qty, timestamp).
        """
        # Do nothing if the list is empty
        if not fills_data:
            return
            
        # The SQL query with placeholders for one row
        q = "INSERT INTO fills(position_id, fill_type, price, qty, ts) VALUES($1, $2, $3, $4, $5)"

        try:
            # executemany sends the query and the entire list of data
            # to the database in a single network round-trip.
            await self.pool.executemany(q, fills_data)
            LOG.info("Successfully batch-inserted %d fill records.", len(fills_data))
        except Exception as e:
            LOG.error("Failed to batch-insert fills: %s", e)




###############################################################################
# 4 ▸ TELEGRAM ###############################################################
###############################################################################

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self._sess: Optional[aiohttp.ClientSession] = None
        self.offset: Optional[int] = None

    async def _req(self, method: str, **params):
        if self._sess is None:
            self._sess = aiohttp.ClientSession()
        async with self._sess.post(f"{self.base}/{method}", json=params) as r:
            return await r.json()

    async def send(self, text: str):
        await self._req("sendMessage", chat_id=self.chat_id, text=text)

    async def poll_cmds(self):
        data = await self._req("getUpdates", offset=self.offset, timeout=0, limit=20)
        for upd in data.get("result", []):
            self.offset = upd["update_id"] + 1
            if (m := upd.get("message")) and (txt := m.get("text")):
                yield txt.strip()


###############################################################################
# 5 ▸ HOT‑RELOAD WATCHER ######################################################
###############################################################################


class _Watcher(FileSystemEventHandler):
    def __init__(self, path: Path, cb):
        self.path = path.resolve()
        self.cb = cb
        obs = Observer()
        obs.schedule(self, self.path.parent.as_posix(), recursive=False)
        obs.daemon = True
        obs.start()

    def on_modified(self, e):  # noqa: N802
        if Path(e.src_path).resolve() == self.path:
            self.cb()


###############################################################################
# 6 ▸ DATA CLASSES ###########################################################
###############################################################################


@dataclasses.dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float
    regime: str


###############################################################################
# 7 ▸ RISK MANAGER ###########################################################
###############################################################################


class RiskManager:
    def __init__(self, cfg_dict: Dict[str, Any]):
        self.cfg = cfg_dict
        self.loss_streak = 0
        self.kill_switch = False

    async def on_trade_close(self, pnl: float, telegram: TelegramBot):
        if pnl < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        if self.loss_streak >= self.cfg["MAX_LOSS_STREAK"]:
            self.kill_switch = True
            await telegram.send("❌ Kill‑switch: max loss streak")

    def can_trade(self) -> bool:
        return not self.kill_switch


###############################################################################
# 8 ▸ MAIN TRADER ###########################################################
###############################################################################


class LiveTrader:
    def __init__(self, settings: Settings, cfg_dict: Dict[str, Any]):
        self.settings = settings
        self.cfg = cfg_dict

        # Propagate YAML overrides into the static config module for older code
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)

        self.db = DB(settings.pg_dsn)
        self.tg = TelegramBot(settings.tg_bot_token, settings.tg_chat_id)
        self.risk = RiskManager(cfg_dict)
        self.filter_rt = filters.Runtime()

        self.exchange = ExchangeProxy(self._init_ccxt())

        self.symbols = self._load_symbols()
        self.open_positions: Dict[int, Dict[str, Any]] = {}  # pid → row dict
        self.signal_generators: Dict[str, SignalGenerator] = {}
        
        # keeps last‑exit timestamps {symbol: utc_datetime}
        self.last_exit: Dict[str, datetime] = {}

        # hot‑reload on file edit
        _Watcher(CONFIG_PATH, self._reload_cfg)
        _Watcher(SYMBOLS_PATH, self._reload_symbols)

        self.paused = False
        self.tasks: List[asyncio.Task] = []
        self.api_semaphore = asyncio.Semaphore(20)

    # ---------------- HELPERS -------------------
    def _init_ccxt(self):
        url = (
            "https://api-testnet.bybit.com"
            if self.settings.bybit_testnet
            else "https://api.bybit.com"
        )
        ex = ccxt.bybit(
            {
                "apiKey": self.settings.bybit_api_key,
                "secret": self.settings.bybit_api_secret,
                "enableRateLimit": True,
            }
        )
        ex.urls["api"] = {"public": url, "private": url}
        return ex

    @staticmethod
    def _load_symbols():
        return SYMBOLS_PATH.read_text().split()

    def _reload_cfg(self):
        self.cfg.update(load_yaml(CONFIG_PATH))
        LOG.info("Config reloaded")

    def _reload_symbols(self):
        self.symbols = self._load_symbols()
        LOG.info("Symbols reloaded – %d symbols", len(self.symbols))

    async def _ensure_leverage(self, symbol: str):
        try:
            await self.exchange.set_margin_mode("CROSSED", symbol)
            await self.exchange.set_leverage(self.settings.default_leverage, symbol)
        except Exception as e:  # noqa: BLE001
            LOG.warning("leverage setup failed %s", e)

    # ------------------------------------------------------------------#
    #                 LISTING‑DATE CACHE (JSON on disk)                 #
    # ------------------------------------------------------------------#
    async def _load_listing_dates(self) -> Dict[str, datetime.date]:
        """
        Returns {symbol: first‑trade‑date}.
        Tries listing_dates.json first; otherwise queries exchange once and caches.
        """
        if LISTING_PATH.exists():
            import datetime as _dt
            raw = json.loads(LISTING_PATH.read_text())
            return {s: _dt.date.fromisoformat(ts) for s, ts in raw.items()}

        LOG.info("listing_dates.json not found. Fetching from exchange (one-time operation)...")

        async def fetch_date(sym):
            try:
                # Fetch the very first daily candle to get the listing date
                candles = await self.exchange.fetch_ohlcv(sym, timeframe="1d", limit=1, since=0)
                if candles:
                    # CCXT returns timestamp in milliseconds
                    ts = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
                    return sym, ts.date()
            except Exception as e:
                LOG.warning("Could not fetch listing date for %s: %s", sym, e)
            return sym, None

        tasks = [fetch_date(sym) for sym in self.symbols]
        results = await asyncio.gather(*tasks)
        
        out = {sym: dt for sym, dt in results if dt}

        # Cache the results to disk to avoid this on next startup
        LISTING_PATH.write_text(
            json.dumps({k: v.isoformat() for k, v in out.items()}, indent=2)
        )
        LOG.info("Saved %d listing dates to %s", len(out), LISTING_PATH)
        return out

    async def _fetch_platform_balance(self) -> dict:
        """
        Fetches balance from the exchange, using the correct method
        based on the configured account type (STANDARD vs UNIFIED).
        """
        account_type = self.cfg.get("BYBIT_ACCOUNT_TYPE", "STANDARD").upper()
        params = {}
        if account_type == "UNIFIED":
            params['accountType'] = 'UNIFIED'
        
        try:
            return await self.exchange.fetch_balance(params=params)
        except Exception as e:
            LOG.error("Failed to fetch %s account balance: %s", account_type, e)
            # Return a zeroed-out structure on failure to prevent crashes downstream
            return {"total": {"USDT": 0.0}, "free": {"USDT": 0.0}}

    # ---------------- POSITION SIZING ---------------
    async def _risk_amount(self, free_usdt: float) -> float:
        mode = self.cfg.get("RISK_MODE", "PERCENT").upper()
        if mode == "PERCENT":
            return free_usdt * self.cfg["RISK_PCT"]
        return float(self.cfg["FIXED_RISK_USDT"])


    # ────────────────────────────────────────────────────────────────────────
    #   Quick ATR(14, 1h) helper with 2‑minute cache
    # ────────────────────────────────────────────────────────────────────────

    # ---------------- ORDER TAGS ------------------
    @staticmethod
    def _cid(pid: int, tag: str) -> str:
        return f"bot_{pid}_{tag}"

    # ---------------- PLACE NEW POSITION ----------
    async def _open_position(self, sig: Signal):
        # skip if already in trade or cooling‑down
        if any(row["symbol"] == sig.symbol for row in self.open_positions.values()):
            return

        cd_h = self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS)
        last_x = self.last_exit.get(sig.symbol)
        if last_x and datetime.now(timezone.utc) - last_x < timedelta(hours=cd_h): # MODIFIED
            LOG.debug("Cool‑down veto on %s (%.1f h left)", sig.symbol,
                    (timedelta(hours=cd_h) - (datetime.now(timezone.utc) - last_x)).total_seconds() / 3600) # MODIFIED
            return

        bal = await self._fetch_platform_balance()
        free_usdt = bal["USDT"]["free"]
        risk_amt = await self._risk_amount(free_usdt)

        entry = sig.entry
        atr   = sig.atr

        stop = entry + self.cfg["SL_ATR_MULT"] * atr
        stop_dist = abs(entry - stop)

        # EXEC‑ATR veto (same limits as back‑tester)
        if (
            stop_dist < self.cfg["MIN_STOP_DIST_USD"]
            or (stop_dist / entry) < self.cfg["MIN_STOP_DIST_PCT"]
        ):
            LOG.info("EXEC_ATR veto %s – stop %.4f < min dist", sig.symbol, stop_dist)
            return

        size = risk_amt / stop_dist

        # --- NEW: Round size to exchange precision and check min cost ---
        try:
            market = self.exchange.markets[sig.symbol]
            
            # 1. Round the amount to the exchange's required precision (step size)
            size = self.exchange.amount_to_precision(sig.symbol, size)
            
            # 2. Check if the order meets the minimum cost (minNotional)
            min_cost = market.get('limits', {}).get('cost', {}).get('min', 0.01)
            order_cost = size * entry
            
            if order_cost < min_cost:
                LOG.warning(
                    "VETO: Order for %s of size %s costs %.4f USDT, "
                    "but exchange requires at least %.4f USDT. Skipping.",
                    sig.symbol, size, order_cost, min_cost
                )
                return

        except Exception as e:
            LOG.error("Error during size validation for %s: %s", sig.symbol, e)
            return
        # --- END OF NEW LOGIC ---

        # time‑based exit deadline
        exit_deadline = (
            datetime.now(timezone.utc) + timedelta(days=self.cfg.get("TIME_EXIT_DAYS", cfg.TIME_EXIT_DAYS)) # MODIFIED
            if self.cfg.get("TIME_EXIT_ENABLED", cfg.TIME_EXIT_ENABLED)
            else None
        )

        now = datetime.now(timezone.utc)
        pid = await self.db.insert_position({
            "symbol": sig.symbol,
            "side": "short",
            "size": size,
            "entry_price": entry,
            "stop_price": stop,
            "trailing_active": False,
            "atr": atr,
            "status": "PENDING",
            "opened_at": now,
            "exit_deadline": exit_deadline,
        })

        try:
            await self._ensure_leverage(sig.symbol)
            await self.exchange.create_market_order(
                sig.symbol, "sell", size,
                params={"clientOrderId": self._cid(pid, "ENTRY")}
            )
            # initial SL
            await self.exchange.create_order(
                sig.symbol, "STOP_MARKET", "buy", size,
                params={"stopPrice": stop, "clientOrderId": self._cid(pid, "SL"), 'reduceOnly': True}
            )
            # partial TP
            if self.cfg.get("PARTIAL_TP_ENABLED", True):
                tp1 = entry - self.cfg["PARTIAL_TP_ATR_MULT"] * atr
                qty = size * self.cfg["PARTIAL_TP_PCT"]
                await self.exchange.create_order(
                    sig.symbol, "TAKE_PROFIT_MARKET", "buy", qty,
                    params={"triggerPrice": tp1, "clientOrderId": self._cid(pid, "TP1"), 'reduceOnly': True}
                )
            await self.db.update_position(pid, status="OPEN")
            row = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)
            self.open_positions[pid] = dict(row)
            await self.tg.send(f"🚀 Opened {sig.symbol} short {size:.3f} @ {entry:.4f}")
        except Exception as e:  # noqa: BLE001
            LOG.error("order placement failed %s", e)
            await self.db.update_position(pid, status="ERROR")

    # ---------------- MANAGE OPEN POSITIONS -------
    async def _manage_positions_loop(self):
        while True:
            if not self.open_positions:
                await asyncio.sleep(2)
                continue
            for pid, pos in list(self.open_positions.items()):
                symbol = pos["symbol"]
                try:
                    await self._update_single_position(pid, pos)
                except Exception as e:  # noqa: BLE001
                    LOG.error("manage err %s %s", symbol, e)
            await asyncio.sleep(5)

    async def _update_single_position(self, pid: int, pos: Dict[str, Any]):
        is_closed = False
        symbol = pos["symbol"]
        orders = await self.exchange.fetch_open_orders(symbol)
        open_cids = {o.get("clientOrderId") for o in orders}

        # ── TIME‑BASED HARD EXIT ──────────────────────────────────────────
        if self.cfg.get("TIME_EXIT_ENABLED", cfg.TIME_EXIT_ENABLED):
            ddl = pos.get("exit_deadline")
            if ddl and datetime.now(timezone.utc) >= ddl: # MODIFIED
                LOG.info("Time‑exit firing on %s (pid %d)", symbol, pid)
                await self._force_close_position(pid, pos, tag="TIME_EXIT")
                return

        # Detect partial TP fill ------------------------------------------------
        if (not pos["trailing_active"]) and self._cid(pid, "TP1") not in open_cids:
            # TP1 gone ⇒ filled
            fill_price = None
            try:
                o = await self.exchange.fetch_order_by_client_id(self._cid(pid, "TP1"), symbol)
                fill_price = o.get("average") or o.get("price")
            except Exception as e:  # noqa: BLE001
                LOG.warning("Failed to fetch TP1 fill price %s", e)

            await self.db.add_fill(
                pid, "TP1", fill_price, pos["size"] * self.cfg["PARTIAL_TP_PCT"]
            )
            await self.db.update_position(pid, trailing_active=True)
            pos["trailing_active"] = True
            # Cancel old SL and place first trailing stop
            await self._activate_trailing(pid, pos)
            await self.tg.send(f"📈 TP1 hit on {symbol}, trailing activated")

        # If trailing active : tighten stop each loop --------------------------
        if pos["trailing_active"]:
            await self._trail_stop(pid, pos)

        # Check if the active stop order is gone -------------------------------
        active_stop_cid = (
            self._cid(pid, "SL_TRAIL") if pos["trailing_active"] else self._cid(pid, "SL")
        )
        if active_stop_cid not in open_cids:
            # The active stop-loss was hit
            is_closed = True
        elif pos["trailing_active"] and self._cid(pid, "TP2") not in open_cids:
            # The final take-profit (TP2) was hit
            is_closed = True
            
        if is_closed:
            await self._finalize_position(pid, pos)

    async def _activate_trailing(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        
        try:
            await self.exchange.cancel_order_by_client_id(self._cid(pid, "SL"), symbol)
        except ccxt.OrderNotFound:
            pass 
        except Exception as e:
            LOG.warning(f"Could not cancel original SL for {pid}: {e}")


        await self._trail_stop(pid, pos, first=True)


        if self.cfg.get("FINAL_TP_ENABLED", False):
            try:

                final_tp_price = pos["entry_price"] - self.cfg["FINAL_TP_ATR_MULT"] * pos["atr"]
                
                qty_left = pos["size"] * (1 - self.cfg["PARTIAL_TP_PCT"])

                LOG.info(f"Placing final TP for {symbol} (pos {pid}) at {final_tp_price:.4f}")

                await self.exchange.create_order(
                    symbol, "TAKE_PROFIT_MARKET", "buy", qty_left,
                    params={"triggerPrice": final_tp_price, "clientOrderId": self._cid(pid, "TP2"), 'reduceOnly': True}
                )                

            except Exception as e:
                LOG.error(f"Failed to place final TP2 order for {pid}: {e}")

    async def _trail_stop(self, pid: int, pos: Dict[str, Any], first: bool = False):
        symbol = pos["symbol"]
        price = (await self.exchange.fetch_ticker(symbol))["last"]
        atr = pos["atr"]

        new_stop = price - self.cfg.get("TRAIL_DISTANCE_ATR_MULT", 1.0) * atr
        is_favorable_move = new_stop < pos["stop_price"]
        min_move_required = pos["stop_price"] * self.cfg.get("TRAIL_MIN_MOVE_PCT", 0.001)
        is_significant_move = (pos["stop_price"] - new_stop) > min_move_required

        if first or (is_favorable_move and is_significant_move):
            LOG.info(
                "Trailing stop for %s. Old: %.4f, New: %.4f", 
                symbol, pos["stop_price"], new_stop
            )
            qty_left = pos["size"] * (1 - self.cfg.get("PARTIAL_TP_PCT", 0.7))
        
            # cancel existing trail order if any
            try:
                await self.exchange.cancel_order_by_client_id(self._cid(pid, "SL_TRAIL"), symbol)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Cancel existing trail failed %s", e)
                # If the order still exists, avoid creating a duplicate
                orders = await self.exchange.fetch_open_orders(symbol)
                if any(o.get("clientOrderId") == self._cid(pid, "SL_TRAIL") for o in orders):
                    return
            await self.exchange.create_order(
                symbol, "STOP_MARKET", "buy", qty_left,
                params={"stopPrice": new_stop, "clientOrderId": self._cid(pid, "SL_TRAIL"), 'reduceOnly': True}
            )
            await self.db.update_position(pid, stop_price=new_stop)
            pos["stop_price"] = new_stop
            LOG.info("Trail updated %s to %.4f", symbol, new_stop)


    async def _finalize_position(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        exit_price = None
        exit_qty = 0.0
        closing_order_type = "UNKNOWN_CLOSE"

        # Determine which order closed the position and get its fill price
        active_stop_cid = self._cid(pid, "SL_TRAIL") if pos["trailing_active"] else self._cid(pid, "SL")
        cids_to_check = [active_stop_cid]
        if pos["trailing_active"] and self.cfg.get("FINAL_TP_ENABLED", False):
            cids_to_check.append(self._cid(pid, "TP2"))

        for cid in cids_to_check:
            try:
                order = await self.exchange.fetch_order_by_client_id(cid, symbol)
                # Check if the order was filled
                if order and order.get('status') == 'closed' and order.get('filled', 0) > 0:
                    exit_price = order.get('average') or order.get('price')
                    exit_qty = order['filled']
                    closing_order_type = "TP2" if "TP2" in cid else "SL"
                    LOG.info(f"Position {pid} closed by order {cid} at avg price {exit_price}")
                    break  # Found the closing order
            except ccxt.OrderNotFound:
                # This is expected if this order wasn't the one that got filled
                continue
            except Exception as e:
                LOG.warning(f"Could not fetch closing order {cid} for PnL calc: {e}")

        # Fallback to ticker price if we couldn't get the fill price
        if not exit_price:
            LOG.warning(f"Could not determine exact fill price for {pid}. Using last market price for PnL.")
            ticker = await self.exchange.fetch_ticker(symbol)
            exit_price = ticker["last"]
            # Estimate exit quantity based on position state
            exit_qty = (
                pos["size"] * (1 - self.cfg.get("PARTIAL_TP_PCT", 0.7))
                if pos["trailing_active"] else pos["size"]
            )

        # Ensure any other related orders are cancelled
        # 1. Define all possible clientOrderIds for this position that might still be open
        cids_to_cleanup = []
        if pos["trailing_active"]:
            # If trailing was active, the final TP (TP2) and the last trail SL might be open
            if self.cfg.get("FINAL_TP_ENABLED", False):
                cids_to_cleanup.append(self._cid(pid, "TP2"))
            cids_to_cleanup.append(self._cid(pid, "SL_TRAIL"))
        else:
            # If it never trailed, the initial SL and TP1 might be open
            cids_to_cleanup.append(self._cid(pid, "SL"))
            if self.cfg.get("PARTIAL_TP_ENABLED", True):
                cids_to_cleanup.append(self._cid(pid, "TP1"))
        
        # 2. Fetch open orders and find the exchange IDs to cancel
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
            order_ids_to_cancel = [
                o['id'] for o in open_orders 
                if o.get('clientOrderId') in cids_to_cleanup
            ]

            # 3. Batch-cancel the identified orders in a single request
            if order_ids_to_cancel:
                LOG.info(f"Batch-cancelling {len(order_ids_to_cancel)} remaining orders for position {pid}.")
                await self.exchange.cancel_orders(order_ids_to_cancel, symbol)

        except Exception as e:
            LOG.warning(f"Precise order cleanup for position {pid} failed: {e}. Falling back to cancel_all_orders for safety.")
            # Fallback to ensure the symbol is clear of any stray orders
            await self.exchange.cancel_all_orders(symbol)

        # Calculate PnL with the best available price and finalize in DB
        pnl = (pos["entry_price"] - exit_price) * exit_qty
        await self.db.update_position(
            pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=pnl
        )
        await self.db.add_fill(pid, closing_order_type, exit_price, exit_qty)
        await self.risk.on_trade_close(pnl, self.tg)
        
        del self.open_positions[pid]
        self.last_exit[symbol] = datetime.now(timezone.utc)
        await self.tg.send(f"✅ {symbol} position closed. PnL ≈ {pnl:.2f} USDT")


    async def _force_close_position(self, pid: int, pos: Dict[str, Any], tag: str):
        """Market‑closes the remaining leg, cancels all orders, finalises DB."""
        symbol = pos["symbol"]
        try:
            await self.exchange.cancel_all_orders(symbol)
            await self.exchange.create_market_order(symbol, "buy", pos["size"])
        except Exception as e:
            LOG.warning("Force‑close order issue on %s: %s", symbol, e)

        last_price = (await self.exchange.fetch_ticker(symbol))["last"]
        pnl = (pos["entry_price"] - last_price) * pos["size"]
        await self.db.update_position(
            pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=pnl
        )
        await self.db.add_fill(pid, tag, last_price, pos["size"])
        await self.risk.on_trade_close(pnl, self.tg)
        self.last_exit[symbol] = datetime.now(timezone.utc)
        del self.open_positions[pid]
        await self.tg.send(f"⏰ {symbol} closed by {tag}. PnL ≈ {pnl:.2f} USDT")

    async def _fetch_and_process_symbol(self, sym: str):
        """
        A concurrent worker to fetch data and process signals for one symbol.
        Uses the class semaphore to limit concurrent exchange requests.
        """
        generator = self.signal_generators.get(sym)
        if not generator or not generator.is_warmed_up:
            return # Skip if not ready

        try:
            async with self.api_semaphore:
                # Fetch the most recent 2 candles
                ohlcvs = await self.exchange.fetch_ohlcv(sym, '4h', limit=2)
            
            if not ohlcvs or len(ohlcvs) < 2:
                return

            last_closed_candle = ohlcvs[0]

            close_price = last_closed_candle[4]
            volume = last_closed_candle[5]
            self.filter_rt.update_ticker(sym, close_price, volume)

            # Let the generator process it and check for a signal
            signal = generator.update_and_check(last_closed_candle)

            if signal:
                # We have a signal, now run it through the existing filters
                equity = await self.db.latest_equity() or 0.0
                ok, vetoes = filters.evaluate(
                    signal,
                    self.filter_rt,
                    open_positions=len(self.open_positions),
                    equity=equity,
                )
                if ok:
                    await self._open_position(signal)
                else:
                    LOG.info("Signal for %s vetoed: %s", sym, " | ".join(vetoes))

        except Exception as e:
            LOG.error("Error processing symbol %s: %s", sym, e)

    # ---------------- MAIN SIGNAL LOOP -----------------
    async def _main_signal_loop(self):
        """
        Central loop that concurrently polls for new candles, updates generators,
        and triggers trades.
        """
        # --- One-time setup: Instantiate and warm up a generator for each symbol ---
        for sym in self.symbols:
            gen = SignalGenerator(sym, self.exchange)
            self.signal_generators[sym] = gen
            asyncio.create_task(gen.warm_up())
        
        await asyncio.sleep(15)
        LOG.info("All signal generators warmed up. Starting main scan loop.")

        # --- The main concurrent polling logic ---
        while True:
            try:
                if self.paused or not self.risk.can_trade():
                    await asyncio.sleep(5)
                    continue

                # 1. Create a list of tasks, one for each symbol.
                #    This does NOT run them yet.
                tasks = [self._fetch_and_process_symbol(sym) for sym in self.symbols]
                
                # 2. Run all tasks concurrently and wait for them all to complete.
                #    This is the key to the performance improvement.
                await asyncio.gather(*tasks)

            except Exception as e:
                LOG.error("Error in main signal loop: %s", e)

            # Wait for a configurable interval before the next full scan.
            await asyncio.sleep(self.cfg.get("SCAN_INTERVAL_SEC", 60))

    # ---------------- EQUITY SNAPSHOT ----------
    async def _equity_loop(self):
        while True:
            try:
                bal = await self._fetch_platform_balance()
                current_equity = bal["total"]["USDT"]
                await self.db.snapshot_equity(current_equity)

                # --- Drawdown Kill-Switch Logic ---
                if self.cfg.get("DD_PAUSE_ENABLED", True) and not self.risk.kill_switch:
                    # Find the peak equity from our own records
                    peak_equity = await self.db.pool.fetchval(
                        "SELECT MAX(equity) FROM equity_snapshots"
                    )
                    if peak_equity and current_equity < peak_equity:
                        drawdown_pct = (peak_equity - current_equity) / peak_equity * 100
                        max_dd_pct = self.cfg.get("DD_MAX_PCT", 10.0)

                        if drawdown_pct >= max_dd_pct:
                            self.risk.kill_switch = True
                            msg = (
                                f"❌ KILL-SWITCH ACTIVATED: "
                                f"Equity drawdown of {drawdown_pct:.2f}% exceeded threshold of {max_dd_pct}%. "
                                f"Peak: ${peak_equity:,.2f}, Current: ${current_equity:,.2f}."
                            )
                            LOG.warning(msg)
                            await self.tg.send(msg)
            
            except Exception as e:
                LOG.error("Error in equity loop: %s", e)

            await asyncio.sleep(3600) # Runs every hour

    # ---------------- CMD LOOP ------------------
    async def _telegram_loop(self):
        while True:
            async for cmd in self.tg.poll_cmds():
                await self._handle_cmd(cmd)
            await asyncio.sleep(1)

    async def _handle_cmd(self, cmd: str):
        parts = cmd.split()
        root = parts[0].lower()
        if root == "/pause":
            self.paused = True
            await self.tg.send("⏸ Paused")
        elif root == "/resume":
            if self.risk.can_trade():
                self.paused = False
                await self.tg.send("▶️ Resumed")
            else:
                await self.tg.send("⚠️ Kill switch active")
        elif root == "/set" and len(parts) == 3:
            key, val = parts[1], parts[2]
            try:
                cast = json.loads(val)
            except json.JSONDecodeError:
                cast = val
            self.cfg[key] = cast
            await self.tg.send(f"✅ {key} set to {cast}")
        elif root == "/status":
            await self.tg.send(json.dumps({
                "paused": self.paused,
                "open": len(self.open_positions),
                "loss_streak": self.risk.loss_streak
            }, indent=2))

    # ---------------- RESUME STATE -------------
    async def _resume(self):
        """
        On startup, load state from DB and reconcile with actual
        exchange positions to detect and handle orphans.
        """
        LOG.info("Resuming state and reconciling positions...")
    

        # 1. Fetch all open positions from the exchange
        try:
            exchange_positions = await self.exchange.fetch_positions()
            # Filter for positions with a non-zero size
            open_exchange_positions = {
                p['info']['symbol']: p for p in exchange_positions if float(p['info']['size']) > 0
            }
        except Exception as e:
            LOG.error("Could not fetch exchange positions on startup: %s. Exiting.", e)
            # This is critical, so we should probably not continue.
            sys.exit(1)

        # 2. Fetch all 'OPEN' positions from our database
        db_positions = {r["symbol"]: dict(r) for r in await self.db.fetch_open_positions()}
        
        # 3. Reconcile
        LOG.info(
            "Found %d positions on exchange and %d 'OPEN' positions in DB.",
            len(open_exchange_positions), len(db_positions)
        )

        # Case A: Position exists on exchange but NOT in our DB (Orphan)
        # This is the most dangerous case. Close it immediately.
        for symbol, pos_data in open_exchange_positions.items():
            if symbol not in db_positions:
                msg = (
                    f"🚨 ORPHAN DETECTED: Position for {symbol} exists on the exchange "
                    f"but not in the database. Forcing immediate close to manage risk."
                )
                LOG.warning(msg)
                await self.tg.send(msg)
                try:
                    # Create a market order to close the position
                    side = 'buy' if pos_data['side'] == 'short' else 'sell'
                    size = float(pos_data['info']['size'])
                    await self.exchange.create_market_order(symbol, side, size, params={'reduceOnly': True})
                except Exception as e:
                    LOG.error("Failed to force-close orphan position %s: %s", symbol, e)

        # Case B: Position exists in our DB but NOT on the exchange
        # This means it was closed while the bot was offline. Mark it as closed.
        for symbol, pos_row in db_positions.items():
            if symbol not in open_exchange_positions:
                pid = pos_row["id"]
                msg = (
                    f"DB/EXCHANGE MISMATCH: Position {pid} for {symbol} is 'OPEN' in DB "
                    f"but not found on exchange. Marking as closed."
                )
                LOG.warning(msg)
                await self.db.update_position(
                    pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=0
                )

        # 4. Load the correctly reconciled positions into memory
        final_open_rows = await self.db.fetch_open_positions()
        for r in final_open_rows:
            self.open_positions[r["id"]] = dict(r)
            LOG.info("Successfully resumed and verified open position for %s (ID: %d)", r["symbol"], r["id"])

        # load last‑exit timestamps within cool‑down window
        cd_h = int(self.cfg.get("SYMBOL_COOLDOWN_HOURS",
                        cfg.SYMBOL_COOLDOWN_HOURS))
        rows = await self.db.pool.fetch(
            "SELECT symbol, closed_at FROM positions "
            "WHERE status='CLOSED' AND closed_at > (NOW() AT TIME ZONE 'utc') - $1::interval",
            timedelta(hours=cd_h),
        )
        for r in rows:
            self.last_exit[r["symbol"]] = r["closed_at"] 

    # ---------------- RUN -----------------------
    async def run(self):
        await self.db.init()
        
        LOG.info("Loading exchange market data...")
        try:
            # Note: We call load_markets directly on the underlying object
            # because it's not a coroutine we want to retry.
            await self.exchange._exchange.load_markets()
            LOG.info("Market data loaded successfully.")
        except Exception as e:
            LOG.error("Could not load exchange market data: %s. Exiting.", e)
            return

        LOG.info("Loading symbol listing dates...")
        listing_dates = await self._load_listing_dates()
        for sym, d in listing_dates.items():
            self.filter_rt.set_listing_date(sym, d)

        await self._resume()
        await self.tg.send("🤖 Bot online v1.2.1")

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._main_signal_loop())
                tg.create_task(self._manage_positions_loop())
                tg.create_task(self._telegram_loop())
                tg.create_task(self._equity_loop())
            
            LOG.info("All tasks finished gracefully.")

        except* (asyncio.CancelledError, KeyboardInterrupt):
            LOG.info("Shutdown signal received. Closing connections...")
        
        finally:
            # This block runs on normal exit or on cancellation.
            await self.exchange.close()
            await self.db.pool.close()
            LOG.info("Bot shut down cleanly.")


###############################################################################
# 9 ▸ ENTRYPOINT ##############################################################
###############################################################################


async def async_main():
    try:
        settings = Settings()
    except ValidationError as e:
        LOG.error("Bad env: %s", e)
        sys.exit(1)

    cfg_dict = load_yaml(CONFIG_PATH)
    trader = LiveTrader(settings, cfg_dict)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(async_main())

# ----------------------------------------------------------------------
# Backward‑compat shim for live_trader.py
# ----------------------------------------------------------------------
try:
    from .scanner import scan_single   # if you moved it
except ImportError:
    scan_single = None  # type: ignore

async def scan_symbol(sym: str, cfg: dict):
    """
    Legacy wrapper so live_trader still calls `scout.scan_symbol`.
    Delegates to `scan_single()` if present.
    """
    if scan_single is None:
        raise NotImplementedError("scan_single() not found in scout module")
    return await scan_single(sym, cfg)