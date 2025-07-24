"""
live_trader.py – v1.2 (CLOSED‑LOOP, bug‑fix release)
====================================================
Changelog
---------
1. Final closure detection now tracks the *active* stop order (SL vs. SL_TRAIL).
2. TP1 fills are logged with the actual average fill price when available.
3. Trail‑stop rotation guards against duplicate SL_TRAIL orders after a
   cancellation error.

No configuration or database schema changes are required vs. v1.1.
"""

from __future__ import annotations

import asyncio

import dataclasses
import json
import logging
import os
import signal as sigmod
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncpg
import ccxt.async_support as ccxt
import yaml
import config as cfg
import filters 
from pydantic import BaseSettings, Field, ValidationError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ---------------------------  YOUR MODULES  ------------------------------
import scout  # async def scan_symbol(sym:str, cfg:dict) -> Optional[Signal]
# -------------------------------------------------------------------------

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
"""

class DB:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def init(self):
        self.pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self.pool.acquire() as conn:
            await conn.execute(TABLES_SQL)

    # ---------- helper wrappers ------------------------------------------
    async def insert_position(self, data: Dict[str, Any]) -> int:
        q = """INSERT INTO positions(symbol,side,size,entry_price,stop_price,trailing_active,atr,status,opened_at)
                 VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id"""
        return await self.pool.fetchval(q, data["symbol"], data["side"], data["size"], data["entry_price"],
                                        data["stop_price"], data["trailing_active"], data["atr"],
                                        data["status"], data["opened_at"])  # type: ignore[index]

    async def update_position(self, pid: int, **fields):
        sets = ",".join(f"{k}=${i+2}" for i, k in enumerate(fields))
        await self.pool.execute(f"UPDATE positions SET {sets} WHERE id=$1", pid, *fields.values())  # type: ignore[index]

    async def add_fill(self, pid: int, fill_type: str, price: Optional[float], qty: float):
        await self.pool.execute(
            "INSERT INTO fills(position_id,fill_type,price,qty,ts) VALUES($1,$2,$3,$4,$5)",
            pid, fill_type, price, qty, datetime.now(timezone.utc)
        )  # type: ignore[index]

    async def fetch_open_positions(self) -> List[asyncpg.Record]:
        return await self.pool.fetch("SELECT * FROM positions WHERE status='OPEN'")  # type: ignore[return-value]

    async def latest_equity(self) -> Optional[float]:
        return await self.pool.fetchval("SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 1")

    async def snapshot_equity(self, equity: float):
        await self.pool.execute(
            "INSERT INTO equity_snapshots VALUES($1,$2) ON CONFLICT DO NOTHING",
            datetime.now(timezone.utc), equity
        )  # type: ignore[index]

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
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
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
    def __init__(self, settings: Settings, cfg: Dict[str, Any]):
        self.settings = settings
        self.cfg = cfg
        for k, v in cfg_yaml.items():
            setattr(cfg, k, v)        
        self.db = DB(settings.pg_dsn)
        self.tg = TelegramBot(settings.tg_bot_token, settings.tg_chat_id)
        self.risk = RiskManager(cfg)
        self.filter_rt = filters.Runtime()

        self.exchange = self._init_ccxt()

        self.symbols = self._load_symbols()
        for sym, d in self._load_listing_dates().items():
            self.filter_rt.set_listing_date(sym, d)       
        self.open_positions: Dict[int, Dict[str, Any]] = {}  # pid -> row dict

        # hot‑reload on file edit
        _Watcher(CONFIG_PATH, self._reload_cfg)
        _Watcher(SYMBOLS_PATH, self._reload_symbols)

        self.paused = False
        self.tasks: List[asyncio.Task] = []

    # ---------------- HELPERS -------------------
    def _init_ccxt(self):
        url = "https://api-testnet.bybit.com" if self.settings.bybit_testnet else "https://api.bybit.com"
        ex = ccxt.bybit({"apiKey": self.settings.bybit_api_key,
                         "secret": self.settings.bybit_api_secret,
                         "enableRateLimit": True})
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

    # ---------------- POSITION SIZING ---------------
    async def _risk_amount(self, free_usdt: float) -> float:
        mode = self.cfg.get("RISK_MODE", "PERCENT").upper()
        if mode == "PERCENT":
            return free_usdt * self.cfg["RISK_PCT"]
        return float(self.cfg["FIXED_RISK_USDT"])

    _atr_cache: Dict[str, Tuple[float, float]] = {}  # symbol → (atr, monotonic_ts)

    async def _fresh_atr(self, symbol: str) -> Optional[float]:
        """Return ATR(14, 1h). Re‑uses a 2‑minute cache to spare API calls."""
        now = asyncio.get_running_loop().time()
        if symbol in self._atr_cache and now - self._atr_cache[symbol][1] < 120:
            return self._atr_cache[symbol][0]
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, "1h", limit=15)
            if len(ohlcv) < 2:
                return None
            import pandas as _pd, indicators as ta
            df = _pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            atr_v = float(ta.atr(df)["atr"].iloc[-1])
            self._atr_cache[symbol] = (atr_v, now)
            return atr_v
        except Exception as e:  # noqa: BLE001
            LOG.warning("ATR fetch failed for %s: %s", symbol, e)
            return None
   
    # ---------------- ORDER TAGS ------------------
    @staticmethod
    def _cid(pid: int, tag: str) -> str:
        return f"bot_{pid}_{tag}"

    # ---------------- PLACE NEW POSITION ----------
    async def _open_position(self, sig: Signal):
        if any(row["symbol"] == sig.symbol for row in self.open_positions.values()):
            return  # already have one

        bal = await self.exchange.fetch_balance()
        free_usdt = bal["USDT"]["free"]
        risk_amt = await self._risk_amount(free_usdt)

        entry = sig.entry
        atr = await self._fresh_atr(sig.symbol) or sig.atr

        stop = entry + self.cfg["SL_ATR_MULT"] * atr
        stop_dist = abs(entry - stop)

        if (
            stop_dist < self.cfg["MIN_STOP_DIST_USD"]
            or (stop_dist / entry) < self.cfg["MIN_STOP_DIST_PCT"]
        ):
            LOG.info("EXEC_ATR veto on %s (stop %.4f too tight)", sig.symbol, stop_dist)
            return

        size = risk_amt / stop_dist

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
                params={"stopPrice": stop, "clientOrderId": self._cid(pid, "SL")}
            )
            # partial TP
            if self.cfg.get("PARTIAL_TP_ENABLED", True):
                tp1 = entry - self.cfg["PARTIAL_TP_ATR_MULT"] * atr
                qty = size * self.cfg["PARTIAL_TP_PCT"]
                await self.exchange.create_order(
                    sig.symbol, "TAKE_PROFIT_MARKET", "buy", qty,
                    params={"triggerPrice": tp1, "clientOrderId": self._cid(pid, "TP1")}
                )
            await self.db.update_position(pid, status="OPEN")
            row = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)  # type: ignore[index]
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
        symbol = pos["symbol"]
        orders = await self.exchange.fetch_open_orders(symbol)
        open_cids = {o.get("clientOrderId") for o in orders}

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
                    params={"triggerPrice": final_tp_price, "clientOrderId": self._cid(pid, "TP2")}
                )
            except Exception as e:
                LOG.error(f"Failed to place final TP2 order for {pid}: {e}")

    async def _trail_stop(self, pid: int, pos: Dict[str, Any], first: bool = False):
        symbol = pos["symbol"]
        price = (await self.exchange.fetch_ticker(symbol))["last"]
        atr = pos["atr"]
        new_stop = price + self.cfg["TRAIL_DISTANCE_ATR_MULT"] * atr
        if first or new_stop < pos["stop_price"]:  # short position ⇒ stop moves DOWN
            qty_left = pos["size"] * (1 - self.cfg["PARTIAL_TP_PCT"])
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
                params={"stopPrice": new_stop, "clientOrderId": self._cid(pid, "SL_TRAIL")}
            )
            await self.db.update_position(pid, stop_price=new_stop)
            pos["stop_price"] = new_stop
            LOG.info("Trail updated %s to %.4f", symbol, new_stop)

    async def _finalize_position(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        
        try:

            if pos["trailing_active"]:
                await self.exchange.cancel_order_by_client_id(self._cid(pid, "TP2"), symbol)
            active_stop_cid = self._cid(pid, "SL_TRAIL") if pos["trailing_active"] else self._cid(pid, "SL")
            await self.exchange.cancel_order_by_client_id(active_stop_cid, symbol)
        except ccxt.OrderNotFound:
            pass
        except Exception as e:
            LOG.warning(f"Could not clean up all orders for closed position {pid}: {e}")


        last_price = exit_price or (await self.exchange.fetch_ticker(symbol))["last"]
        exit_qty = (
            pos["size"] * (1 - self.cfg["PARTIAL_TP_PCT"])
            if pos["trailing_active"] else pos["size"]
        )
        pnl = (pos["entry_price"] - last_price) * exit_qty  # short pnl
        await self.db.update_position(
            pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=pnl
        )
        await self.risk.on_trade_close(pnl, self.tg)
        del self.open_positions[pid]
        await self.tg.send(f"✅ {symbol} position closed. PnL ≈ {pnl:.2f} USDT")

    # ---------------- SIGNAL LOOP -----------------
     async def _signal_loop(self):
         while True:
             try:
                 if self.paused or not self.risk.can_trade():
                     await asyncio.sleep(2)
                     continue

                # fresh account equity for notional / MIN_NOTIONAL checks
                bal = await self.exchange.fetch_balance()
                equity = bal["total"]["USDT"]

                for sym in list(self.symbols):
                     if any(row["symbol"] == sym for row in self.open_positions.values()):
                         continue  # skip; already in trade
                     sig_raw = await scout.scan_symbol(sym, self.cfg)
                    if not sig_raw:
                        continue

                    ok, vetoes = filters.evaluate(
                        sig_raw,
                        self.filter_rt,
                        open_positions=len(self.open_positions),
                        equity=equity,
                    )
                    if not ok:
                        # optional: store veto audit once you add `events` table
                        # await self.db.add_event({...})
                        continue

                        await self._open_position(sig_raw)
                await asyncio.sleep(self.cfg["SCAN_INTERVAL_SEC"])

         self.tasks = [
             asyncio.create_task(self._signal_loop()),
             asyncio.create_task(self._manage_positions_loop()),
             asyncio.create_task(self._telegram_loop()),
             asyncio.create_task(self._equity_loop()),


            # ───────────────────────────────────────────────────────────
            # NEW: background streams feeding the filters.Runtime caches
            # (you will flesh these out in Item 7 when ccxt.pro streams
            # are wired; for now they poll REST every 30 s)
            # ───────────────────────────────────────────────────────────
            asyncio.create_task(self._ticker_feed_loop("BTCUSDT")),
            asyncio.create_task(self._ticker_feed_loop(cfg.ALT_SYMBOL)),
            asyncio.create_task(self._oi_poll_loop()),
         ]
         await asyncio.gather(*self.tasks, return_exceptions=True)

# --------------------------------------------------------------------------- #
# NEW helper coroutines                                                       #
# --------------------------------------------------------------------------- #
    async def _ticker_feed_loop(self, symbol: str):
        """Poll REST every ~30 s until websockets are in place.

        Sends both price **and** per‑interval volume to filters.Runtime
        so it can upkeep the rolling VWAP buffer.
        """
        while True:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price = ticker["last"]

                # Bybit REST doesn’t give true per‑second volume,
                # but we can derive an approximate “bucket” from 24 h data.
                # Good enough for a 5‑min VWAP.
                vol_24h = ticker.get("quoteVolume") or ticker.get("baseVolume")
                vol = vol_24h / 86400 if vol_24h else None  # ≈ per‑sec avg

                self.filter_rt.update_ticker(symbol, price, vol)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Ticker poll failed for %s: %s", symbol, e)

            await asyncio.sleep(30)

            await asyncio.sleep(30)

    async def _oi_poll_loop(self):
        """Rudimentary OI snapshot loop (poll REST every 5 min)."""
        while True:
            try:
                for sym in self.symbols:
                    oi = (await self.exchange.fetch_open_interest(sym))["openInterest"]
                    self.filter_rt.update_open_interest(sym, oi)
            except Exception as e:
                LOG.warning("OI poll failed: %s", e)
            await asyncio.sleep(300)

    # ---------------- EQUITY SNAPSHOT ----------
    async def _equity_loop(self):
        while True:
            bal = await self.exchange.fetch_balance()
            eq = bal["total"]["USDT"]
            await self.db.snapshot_equity(eq)
            await asyncio.sleep(3600)

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
        rows = await self.db.fetch_open_positions()
        for r in rows:
            self.open_positions[r["id"]] = dict(r)
            LOG.info("Resumed open position %s", r["symbol"])

    # ---------------- RUN -----------------------
    async def run(self):
        await self.db.init()
        await self._resume()
        await self.tg.send("🤖 Bot online v1.2")

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(sigmod.SIGINT, lambda: [t.cancel() for t in self.tasks])
        loop.add_signal_handler(sigmod.SIGTERM, lambda: [t.cancel() for t in self.tasks])

        self.tasks = [
            asyncio.create_task(self._signal_loop()),
            asyncio.create_task(self._manage_positions_loop()),
            asyncio.create_task(self._telegram_loop()),
            asyncio.create_task(self._equity_loop()),
        ]
        await asyncio.gather(*self.tasks, return_exceptions=True)

###############################################################################
# 9 ▸ ENTRYPOINT ##############################################################
###############################################################################
async def async_main():
    try:
        settings = Settings()
    except ValidationError as e:
        LOG.error("Bad env: %s", e)
        sys.exit(1)
    cfg = load_yaml(CONFIG_PATH)
    trader = LiveTrader(settings, cfg)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(async_main())

###############################################################################
# TODO – WebSocket order‑stream implementation (ccxt.pro) for instant fills ###
###############################################################################
