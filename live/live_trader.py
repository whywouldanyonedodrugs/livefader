"""
live_trader.py – v3.0 (Simplified & Corrected)
===========================================================================
This version restores the missing _load_listing_dates method that was
accidentally deleted, fixing the startup crash.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import signal as sigmod
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncpg
import ccxt.async_support as ccxt
import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config as cfg
from . import indicators as ta
from . import filters
from . shared_utils import is_blacklisted
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .exchange_proxy import ExchangeProxy
from .database import DB
from .telegram import TelegramBot


from pydantic import Field, Extra, ValidationError
from pydantic_settings import BaseSettings

@dataclasses.dataclass
class Signal:
    symbol: str
    entry: float
    atr: float
    rsi: float
    ret_30d: float
    adx: float
    vwap_consolidated: bool

LISTING_PATH = Path("listing_dates.json")

###############################################################################
# 0 ▸ LOGGING #################################################################
###############################################################################

LOG = logging.getLogger("live_trader")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

###############################################################################
# 1 ▸ SETTINGS pulled from ENV (.env) #########################################
###############################################################################

class Settings(BaseSettings):
    """Secrets & env flags."""
    bybit_api_key: str = Field(..., validation_alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., validation_alias="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(False, validation_alias="BYBIT_TESTNET")
    tg_bot_token: str = Field(..., validation_alias="TG_BOT_TOKEN")
    tg_chat_id: str = Field(..., validation_alias="TG_CHAT_ID")
    pg_dsn: str = Field(..., validation_alias="DATABASE_URL")
    default_leverage: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'

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
# 3 ▸ HOT‑RELOAD WATCHER ######################################################
###############################################################################

class _Watcher(FileSystemEventHandler):
    def __init__(self, path: Path, cb):
        self.path = path.resolve()
        self.cb = cb
        obs = Observer()
        obs.schedule(self, self.path.parent.as_posix(), recursive=False)
        obs.daemon = True
        obs.start()

    def on_modified(self, e):
        if Path(e.src_path).resolve() == self.path:
            self.cb()

###############################################################################
# 4 ▸ RISK MANAGER ###########################################################
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
# 5 ▸ MAIN TRADER ###########################################################
###############################################################################

class LiveTrader:
    def __init__(self, settings: Settings, cfg_dict: Dict[str, Any]):
        self.settings = settings
        self.cfg = {}
        for key in dir(cfg):
            if key.isupper():
                self.cfg[key] = getattr(cfg, key)
        self.cfg.update(cfg_dict)

        for k, v in self.cfg.items():
            setattr(cfg, k, v)

        self.db = DB(settings.pg_dsn)
        self.tg = TelegramBot(settings.tg_bot_token, settings.tg_chat_id)
        self.risk = RiskManager(self.cfg)
        self.exchange = ExchangeProxy(self._init_ccxt())
        self.symbols = self._load_symbols()
        self.open_positions: Dict[int, Dict[str, Any]] = {}
        self.peak_equity: float = 0.0
        self._listing_dates_cache: Dict[str, datetime.date] = {}
        self.last_exit: Dict[str, datetime] = {}
        _Watcher(CONFIG_PATH, self._reload_cfg)
        _Watcher(SYMBOLS_PATH, self._reload_symbols)
        self.paused = False
        self.tasks: List[asyncio.Task] = []
        self.api_semaphore = asyncio.Semaphore(10)
        self.symbol_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _init_ccxt(self):
        ex = ccxt.bybit({
            "apiKey": self.settings.bybit_api_key,
            "secret": self.settings.bybit_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if self.settings.bybit_testnet:
            ex.set_sandbox_mode(True)
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
            await self.exchange.set_margin_mode("cross", symbol)
            await self.exchange.set_leverage(self.settings.default_leverage, symbol)
        except ccxt.ExchangeError as e:
            if "leverage not modified" in str(e):
                LOG.debug("Leverage for %s already set to %dx.", symbol, self.settings.default_leverage)
            else:
                LOG.warning("Leverage setup failed for %s: %s", symbol, e)
                raise e
        except Exception as e:
            LOG.warning("An unexpected error occurred during leverage setup for %s: %s", symbol, e)
            raise e

    # --- THIS METHOD WAS MISSING ---
    async def _load_listing_dates(self) -> Dict[str, datetime.date]:
        if LISTING_PATH.exists():
            import datetime as _dt
            raw = json.loads(LISTING_PATH.read_text())
            return {s: _dt.date.fromisoformat(ts) for s, ts in raw.items()}

        LOG.info("listing_dates.json not found. Fetching from exchange...")
        async def fetch_date(sym):
            try:
                candles = await self.exchange.fetch_ohlcv(sym, timeframe="1d", limit=1, since=0)
                if candles:
                    ts = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
                    return sym, ts.date()
            except Exception as e:
                LOG.warning("Could not fetch listing date for %s: %s", sym, e)
            return sym, None

        tasks = [fetch_date(sym) for sym in self.symbols]
        results = await asyncio.gather(*tasks)
        out = {sym: dt for sym, dt in results if dt}
        LISTING_PATH.write_text(json.dumps({k: v.isoformat() for k, v in out.items()}, indent=2))
        LOG.info("Saved %d listing dates to %s", len(out), LISTING_PATH)
        return out

    async def _fetch_platform_balance(self) -> dict:
        account_type = self.cfg.get("BYBIT_ACCOUNT_TYPE", "STANDARD").upper()
        params = {}
        if account_type == "UNIFIED":
            params['accountType'] = 'UNIFIED'
        try:
            return await self.exchange.fetch_balance(params=params)
        except Exception as e:
            LOG.error("Failed to fetch %s account balance: %s", account_type, e)
            return {"total": {"USDT": 0.0}, "free": {"USDT": 0.0}}

    async def _risk_amount(self, free_usdt: float) -> float:
        mode = self.cfg.get("RISK_MODE", "PERCENT").upper()
        if mode == "PERCENT":
            return free_usdt * self.cfg["RISK_PCT"]
        return float(self.cfg["FIXED_RISK_USDT"])

    @staticmethod
    def _cid(pid: int, tag: str) -> str:
        """
        Creates a unique clientOrderId. Appends a millisecond timestamp
        to the PID and tag to ensure uniqueness across restarts and prevent
        "OrderLinkedID is duplicate" errors.
        """
        # Get the current time in milliseconds
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Create a unique ID that is also short enough for the exchange
        return f"bot_{pid}_{tag}_{timestamp_ms}"

    # --- ADD THIS NEW HELPER METHOD ---
    async def _all_open_orders(self, symbol: str) -> list:
        """
        Fetches all types of open orders (regular, conditional, TP/SL) for a symbol
        by correctly using the Bybit V5 API parameters.
        """
        params_linear = {'category': 'linear'}
        try:
            active_orders = await self.exchange.fetch_open_orders(symbol, params=params_linear)
            stop_orders = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'StopOrder'})
            tpsl_orders = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'tpslOrder'})
            return active_orders + stop_orders + tpsl_orders
        except Exception as e:
            LOG.warning("Could not fetch all open orders for %s: %s", symbol, e)
            return [] # Return an empty list on failure to prevent crashes

    async def _scan_symbol_for_signal(self, symbol: str) -> Optional[Signal]:
        LOG.info("Checking %s...", symbol)
        try:
            base_tf = self.cfg.get('TIMEFRAME', '5m')
            ema_tf = self.cfg.get('EMA_TIMEFRAME', '4h')
            rsi_tf = self.cfg.get('RSI_TIMEFRAME', '1h')
            atr_tf = self.cfg.get('ADX_TIMEFRAME', '1h')
            required_tfs = {base_tf, ema_tf, rsi_tf, atr_tf, '1d'}

            async with self.api_semaphore:
                tasks = {tf: self.exchange.fetch_ohlcv(symbol, tf, limit=500) for tf in required_tfs}
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                ohlcv_data = dict(zip(tasks.keys(), results))

            dfs = {}
            for tf, data in ohlcv_data.items():
                if isinstance(data, Exception) or not data:
                    LOG.debug("Could not fetch OHLCV for %s on %s timeframe.", symbol, tf)
                    return None
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                
                recent_candles = df.tail(100)
                if not recent_candles.empty:
                    zero_volume_pct = (recent_candles['volume'] == 0).sum() / len(recent_candles)
                    if zero_volume_pct > 0.25:
                        LOG.warning("DATA_ERROR for %s on %s: Detected stale data (%.0f%% zero volume). Skipping symbol.", symbol, tf, zero_volume_pct * 100)
                        return None
                
                df.drop(df.index[-1], inplace=True)
                if df.empty: return None
                dfs[tf] = df

            df5 = dfs[base_tf]
            df5['ema_fast'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_FAST_PERIOD).reindex(df5.index, method='ffill')
            df5['ema_slow'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_SLOW_PERIOD).reindex(df5.index, method='ffill')
            df5['rsi'] = ta.rsi(dfs[rsi_tf]['close'], cfg.RSI_PERIOD).reindex(df5.index, method='ffill')
            df5['atr'] = ta.atr(dfs[atr_tf], cfg.ADX_PERIOD).reindex(df5.index, method='ffill')
            df5['adx'] = ta.adx(dfs[atr_tf], cfg.ADX_PERIOD).reindex(df5.index, method='ffill')

            tf_minutes = 5
            boom_bars = int((cfg.PRICE_BOOM_PERIOD_H * 60) / tf_minutes)
            slowdown_bars = int((cfg.PRICE_SLOWDOWN_PERIOD_H * 60) / tf_minutes)
            df5['price_boom_ago'] = df5['close'].shift(boom_bars)
            df5['price_slowdown_ago'] = df5['close'].shift(slowdown_bars)

            df1d = dfs['1d']
            ret_30d = (df1d['close'].iloc[-1] / df1d['close'].iloc[-cfg.STRUCTURAL_TREND_DAYS] - 1) if len(df1d) > cfg.STRUCTURAL_TREND_DAYS else 0.0

            if self.cfg.get("GAP_FILTER_ENABLED", True):
                vwap_bars = int((cfg.GAP_VWAP_HOURS * 60) / tf_minutes)
                vwap_num = (df5['close'] * df5['volume']).shift(1).rolling(vwap_bars).sum()
                vwap_den = df5['volume'].shift(1).rolling(vwap_bars).sum()
                df5['vwap'] = vwap_num / vwap_den
                df5['vwap_dev'] = abs(df5['close'] - df5['vwap']) / df5['vwap']
                df5['vwap_ok'] = df5['vwap_dev'] <= cfg.GAP_MAX_DEV_PCT
                df5['vwap_consolidated'] = df5['vwap_ok'].rolling(cfg.GAP_MIN_BARS).min().fillna(0).astype(bool)
            else:
                df5['vwap_consolidated'] = True
                df5['vwap_dev'] = 0.0
                df5['vwap_ok'] = True
                df5['vwap'] = df5['close']

            df5.dropna(inplace=True)
            if df5.empty: return None

            last = df5.iloc[-1]
            boom_ret_pct = (last['close'] / last['price_boom_ago'] - 1)
            slowdown_ret_pct = (last['close'] / last['price_slowdown_ago'] - 1)
            price_boom = boom_ret_pct > cfg.PRICE_BOOM_PCT
            price_slowdown = slowdown_ret_pct < cfg.PRICE_SLOWDOWN_PCT

            ema_filter_enabled = self.cfg.get("EMA_FILTER_ENABLED", True)
            if ema_filter_enabled:
                ema_down = last['ema_fast'] < last['ema_slow']
                ema_log_msg = f"{'✅' if ema_down else '❌'} (Fast: {last['ema_fast']:.4f} < Slow: {last['ema_slow']:.4f})"
            else:
                ema_down = True
                ema_log_msg = " DISABLED"
            
            trend_filter_enabled = self.cfg.get("STRUCTURAL_TREND_FILTER_ENABLED", True)
            if trend_filter_enabled:
                trend_ok = ret_30d <= self.cfg.get("STRUCTURAL_TREND_RET_PCT", 0.01)
                trend_log_msg = f"{'✅' if trend_ok else '❌'} (Return: {ret_30d:+.2%})"
            else:
                trend_log_msg = " DISABLED"

            gap_filter_enabled = self.cfg.get("GAP_FILTER_ENABLED", True)
            if gap_filter_enabled:
                vwap_dev_pct = last.get('vwap_dev', float('nan'))
                vwap_dev_str = f"{vwap_dev_pct:.2%}" if pd.notna(vwap_dev_pct) else "N/A"
                current_dev_ok = last.get('vwap_ok', False)
                vwap_log_msg = f"{'✅' if last['vwap_consolidated'] else '❌'} (Streak Failed) | Current Dev: {'✅' if current_dev_ok else '❌'} ({vwap_dev_str})"
            else:
                vwap_log_msg = " DISABLED"

            LOG.debug(
                f"\n--- {symbol} | {last.name.strftime('%Y-%m-%d %H:%M')} UTC ---\n"
                f"  [Base Timeframe: {base_tf}]\n"
                f"  - Price Boom     (>{cfg.PRICE_BOOM_PCT:.0%}, {cfg.PRICE_BOOM_PERIOD_H}h lookback): {'✅' if price_boom else '❌'} (is {boom_ret_pct:+.2%})\n"
                f"  - Price Slowdown (<{cfg.PRICE_SLOWDOWN_PCT:.0%}, {cfg.PRICE_SLOWDOWN_PERIOD_H}h lookback): {'✅' if price_slowdown else '❌'} (is {slowdown_ret_pct:+.2%})\n"
                f"  - EMA Trend Down ({ema_tf}):      {ema_log_msg}\n"
                f"  --------------------------------------------------\n"
                f"  - RSI ({rsi_tf}):                 {last['rsi']:.2f} (Veto: {not (cfg.RSI_ENTRY_MIN <= last['rsi'] <= cfg.RSI_ENTRY_MAX)})\n"
                f"  - 30d Trend Filter:        {trend_log_msg}\n"
                f"  - VWAP Consolidated:       {vwap_log_msg}\n"
                f"  - ATR ({atr_tf}):                 {last['atr']:.6f}\n"
                f"====================================================\n"
            )

            if price_boom and price_slowdown and ema_down:
                LOG.info("SIGNAL FOUND for %s at price %.4f", symbol, last['close'])
                return Signal(
                    symbol=symbol, entry=float(last['close']), atr=float(last['atr']),
                    rsi=float(last['rsi']), adx=float(last['adx']), ret_30d=float(ret_30d),
                    vwap_consolidated=bool(last['vwap_consolidated'])
                )
        except ccxt.BadSymbol:
            LOG.warning("Could not scan %s: Invalid symbol on exchange.", symbol)
        except Exception as e:
            LOG.error("Error scanning symbol %s: %s", symbol, e)
            traceback.print_exc()
        return None

    async def _open_position(self, sig: Signal):
        if any(row["symbol"] == sig.symbol for row in self.open_positions.values()):
            return

        cd_h = self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS)
        last_x = self.last_exit.get(sig.symbol)
        if last_x and datetime.now(timezone.utc) - last_x < timedelta(hours=cd_h):
            LOG.debug("Cool-down veto on %s", sig.symbol)
            return

        bal = await self._fetch_platform_balance()
        free_usdt = bal["free"]["USDT"]
        risk_amt = await self._risk_amount(free_usdt)

        entry = sig.entry
        atr = sig.atr
        stop = entry + self.cfg["SL_ATR_MULT"] * atr
        slippage_buffer = entry * self.cfg.get("SLIPPAGE_BUFFER_PCT", 0.0)
        effective_entry_price = entry + slippage_buffer
        stop_dist = abs(effective_entry_price - stop)

        if stop_dist == 0:
            LOG.warning("VETO: Stop distance is zero for %s. Skipping.", sig.symbol)
            return

        if (stop_dist < self.cfg["MIN_STOP_DIST_USD"] or (stop_dist / entry) < self.cfg["MIN_STOP_DIST_PCT"]):
            LOG.info("EXEC_ATR veto %s – stop %.4f < min dist", sig.symbol, stop_dist)
            return

        intended_size = risk_amt / stop_dist

        try:
            market = self.exchange.market(sig.symbol)
            min_cost = market.get('limits', {}).get('cost', {}).get('min')
            if min_cost is None:
                min_cost = self.cfg.get("MIN_NOTIONAL", 0.01)

            if (intended_size * entry) < min_cost:
                LOG.warning("VETO: Order for %s size %s costs less than exchange minimum of %s.", sig.symbol, intended_size, min_cost)
                return
        except Exception as e:
            LOG.error("Error during size validation for %s: %s", sig.symbol, e)
            traceback.print_exc()
            return

        try:
            await self._ensure_leverage(sig.symbol)
        except Exception:
            return

        exit_deadline = (
            datetime.now(timezone.utc) + timedelta(days=self.cfg.get("TIME_EXIT_DAYS", 10))
            if self.cfg.get("TIME_EXIT_ENABLED", True) else None
        )

        pid = await self.db.insert_position({
            "symbol": sig.symbol, "side": "short", "size": intended_size,
            "entry_price": entry, "stop_price": stop, "trailing_active": False,
            "atr": atr, "status": "PENDING", "opened_at": datetime.now(timezone.utc),
            "exit_deadline": exit_deadline,
        })

        entry_order = None
        executed_size = 0.0
        try:
            entry_order = await self.exchange.create_market_order(
                sig.symbol, "sell", intended_size, params={"clientOrderId": self._cid(pid, "ENTRY")}
            )
            
            # --- FINAL, ROBUST CONFIRMATION LOOP ---
            # Trust the initial response if it gives a fill amount.
            executed_size = entry_order.get('filled', 0.0)

            if executed_size <= 0:
                LOG.info("Initial response showed no fill, starting confirmation loop for %s...", sig.symbol)
                for i in range(10):
                    await asyncio.sleep(0.5)
                    order_status = await self.exchange.fetch_order(entry_order['id'], sig.symbol)
                    
                    # This is the critical fix: check if order_status is not None
                    if order_status and order_status.get('status') == 'closed' and order_status.get('filled', 0) > 0:
                        executed_size = order_status.get('filled')
                        LOG.info("Confirmation loop successful. Executed size: %s", executed_size)
                        break
            
            if executed_size <= 0:
                raise ccxt.ExchangeError(f"Entry order for {sig.symbol} was not filled after 5 seconds.")

        except Exception as e:
            LOG.error("ENTRY FAILED for %s (pid %d): %s. Position will not be opened.", sig.symbol, pid, e)
            await self.db.update_position(pid, status="ERROR_ENTRY")
            if entry_order and entry_order.get('id'):
                try:
                    await self.exchange.cancel_order(entry_order['id'], sig.symbol)
                except Exception as cancel_e:
                    LOG.warning("Could not cancel potentially stuck entry order for %s: %s", sig.symbol, cancel_e)
            return

        try:
            sl_params = {"triggerPrice": stop, "clientOrderId": self._cid(pid, "SL"), 'reduceOnly': True, 'triggerDirection': 1}
            await self.exchange.create_order(sig.symbol, 'market', "buy", executed_size, params=sl_params)
            
            if self.cfg.get("PARTIAL_TP_ENABLED", False):
                tp_price = entry - self.cfg["PARTIAL_TP_ATR_MULT"] * atr
                qty = executed_size * self.cfg["PARTIAL_TP_PCT"]
                tp_params = {"triggerPrice": tp_price, "clientOrderId": self._cid(pid, "TP1"), 'reduceOnly': True, 'triggerDirection': 2}
                await self.exchange.create_order(sig.symbol, 'market', "buy", qty, params=tp_params)
            
            elif self.cfg.get("FINAL_TP_ENABLED", False):
                tp_price = entry - self.cfg["FINAL_TP_ATR_MULT"] * atr
                qty = executed_size
                tp_params = {"triggerPrice": tp_price, "clientOrderId": self._cid(pid, "TP_FINAL"), 'reduceOnly': True, 'triggerDirection': 2}
                await self.exchange.create_order(sig.symbol, 'market', "buy", qty, params=tp_params)
            
            await self.db.update_position(pid, status="OPEN", size=executed_size)
            row = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)
            self.open_positions[pid] = dict(row)
            await self.tg.send(f"🚀 Opened {sig.symbol} short {executed_size:.3f} @ {entry:.4f}")

        except Exception as e:
            LOG.critical("NAKED POSITION: SL/TP placement failed for %s (pid %d): %s", sig.symbol, pid, e)
            await self.db.update_position(pid, status="ERROR_NAKED", size=executed_size)
            try:
                await self.exchange.create_market_order(
                    sig.symbol, 'buy', executed_size, params={'reduceOnly': True}
                )
                msg = f"🚨 CRITICAL: Failed to set SL/TP for {sig.symbol}. Position has been emergency closed."
                LOG.warning(msg)
                await self.tg.send(msg)
            except Exception as close_e:
                msg = f"🚨 !!! NAKED POSITION for {sig.symbol}. Manual intervention REQUIRED. Close error: {close_e}"
                LOG.critical(msg)
                await self.tg.send(msg)

    async def _manage_positions_loop(self):
        while True:
            if not self.open_positions:
                await asyncio.sleep(2)
                continue
            for pid, pos in list(self.open_positions.items()):
                try:
                    await self._update_single_position(pid, pos)
                except Exception as e:
                    LOG.error("manage err %s %s", pos["symbol"], e)
            await asyncio.sleep(5)

    async def _update_single_position(self, pid: int, pos: Dict[str, Any]):
        is_closed = False
        symbol = pos["symbol"]
        # --- USE THE NEW HELPER METHOD HERE ---
        orders = await self._all_open_orders(symbol)
        open_cids = {o.get("clientOrderId") for o in orders}

        if self.cfg.get("TIME_EXIT_ENABLED", cfg.TIME_EXIT_ENABLED):
            ddl = pos.get("exit_deadline")
            if ddl and datetime.now(timezone.utc) >= ddl:
                LOG.info("Time-exit firing on %s (pid %d)", symbol, pid)
                await self._force_close_position(pid, pos, tag="TIME_EXIT")
                return

        if self.cfg.get("PARTIAL_TP_ENABLED", False) and (not pos["trailing_active"]) and self._cid(pid, "TP1") not in open_cids:
            fill_price = None
            try:
                o = await self.exchange.fetch_order_by_client_id(self._cid(pid, "TP1"), symbol)
                fill_price = o.get("average") or o.get("price")
            except Exception as e:
                LOG.warning("Failed to fetch TP1 fill price %s", e)

            await self.db.add_fill(
                pid, "TP1", fill_price, pos["size"] * self.cfg["PARTIAL_TP_PCT"], datetime.now(timezone.utc)
            )
            await self.db.update_position(pid, trailing_active=True)
            pos["trailing_active"] = True
            await self._activate_trailing(pid, pos)
            await self.tg.send(f"📈 TP1 hit on {symbol}, trailing activated")

        if pos["trailing_active"]:
            await self._trail_stop(pid, pos)

        active_stop_cid = self._cid(pid, "SL_TRAIL") if pos["trailing_active"] else self._cid(pid, "SL")
        is_closed = active_stop_cid not in open_cids
        if not is_closed and pos["trailing_active"] and self.cfg.get("FINAL_TP_ENABLED", False):
             if self._cid(pid, "TP2") not in open_cids:
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
        min_move_required = price * self.cfg.get("TRAIL_MIN_MOVE_PCT", 0.001)
        is_significant_move = (pos["stop_price"] - new_stop) > min_move_required

        if first or (is_favorable_move and is_significant_move):
            qty_left = pos["size"] * (1 - self.cfg.get("PARTIAL_TP_PCT", 0.7))
            try:
                await self.exchange.cancel_order_by_client_id(self._cid(pid, "SL_TRAIL"), symbol)
            except ccxt.OrderNotFound:
                pass
            except Exception as e:
                LOG.warning("Trail cancel failed for %s: %s. Will retry.", symbol, e)
                return
            
            # Correctly create a separate BUY order, conditional on the triggerPrice (your new SL)
            sl_params = {"triggerPrice": new_stop, "clientOrderId": self._cid(pid, "SL_TRAIL"), 'reduceOnly': True, 'triggerDirection': 1}
            await self.exchange.create_order(
                symbol, 'market', "buy", qty_left, params=sl_params
            )
            await self.db.update_position(pid, stop_price=new_stop)
            pos["stop_price"] = new_stop
            LOG.info("Trail updated %s to %.4f", symbol, new_stop)
            
            # --- BUG FIX: Use 'market' as the order type ---
            await self.exchange.create_order(
                symbol, 'market', "buy", qty_left,
                params={"stopPrice": new_stop, "clientOrderId": self._cid(pid, "SL_TRAIL"), 'reduceOnly': True, 'triggerDirection': 1}
            )
            await self.db.update_position(pid, stop_price=new_stop)
            pos["stop_price"] = new_stop
            LOG.info("Trail updated %s to %.4f", symbol, new_stop)
            
    async def _finalize_position(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        exit_price, exit_qty, closing_order_type = None, 0.0, "UNKNOWN_CLOSE"

        # --- Determine Exit Price & Reason ---
        # This logic is still valuable for accurate PnL reporting.
        # We try to find the exact fill data of the order that closed the position.
        
        # Define all possible order IDs that could have closed the trade
        possible_closing_cids = []
        if pos["trailing_active"]:
            possible_closing_cids.append(self._cid(pid, "SL_TRAIL"))
            if self.cfg.get("FINAL_TP_ENABLED", False):
                possible_closing_cids.append(self._cid(pid, "TP2"))
        else:
            possible_closing_cids.append(self._cid(pid, "SL"))
            if self.cfg.get("PARTIAL_TP_ENABLED", False):
                possible_closing_cids.append(self._cid(pid, "TP1"))
            elif self.cfg.get("FINAL_TP_ENABLED", False):
                possible_closing_cids.append(self._cid(pid, "TP_FINAL"))

        # Check the history for each possible closing order
        for cid in possible_closing_cids:
            try:
                # fetch_order can find closed/filled orders
                order = await self.exchange.fetch_order(clientOrderId=cid, symbol=symbol)
                if order and order.get('status') == 'closed' and order.get('filled', 0) > 0:
                    exit_price = order.get('average') or order.get('price')
                    exit_qty = order['filled']
                    # Determine the closing reason from the CID
                    if "TP" in cid: closing_order_type = "TP"
                    elif "SL" in cid: closing_order_type = "SL"
                    LOG.info(f"Position {pid} closed by order {cid} at avg price {exit_price}")
                    break 
            except ccxt.OrderNotFound:
                continue
            except Exception as e:
                LOG.warning(f"Could not fetch closing order {cid} for PnL calc: {e}")

        # Fallback to ticker price if we couldn't get the exact fill price
        if not exit_price:
            LOG.warning(f"Could not determine exact fill price for {pid}. Using last market price for PnL.")
            ticker = await self.exchange.fetch_ticker(symbol)
            exit_price = ticker["last"]
            # Estimate exit quantity based on position state
            exit_qty = (
                pos["size"] * (1 - self.cfg.get("PARTIAL_TP_PCT", 0.7))
                if pos["trailing_active"] else pos["size"]
            )

        # --- Bullet-Proof Cleanup Protocol ---
        # Unconditionally cancel ALL orders for this symbol to ensure a clean slate.
        try:
            LOG.info("Finalizing position %d for %s. Cancelling all related orders.", pid, symbol)
            await self.exchange.cancel_all_orders(symbol, params={'category': 'linear'})
        except Exception as e:
            LOG.warning(f"Final cleanup for position {pid} failed: {e}. The exchange's reduceOnly should prevent issues.")
        
        # --- Database and State Update ---
        # Record the final fill that closed the position
        await self.db.add_fill(pid, closing_order_type, exit_price, exit_qty, datetime.now(timezone.utc))

        # Fetch ALL fills for this position to calculate total PnL accurately
        all_fills = await self.db.pool.fetch("SELECT price, qty FROM fills WHERE position_id=$1", pid)
        total_pnl = 0
        entry_price = pos["entry_price"]
        for fill in all_fills:
            if fill['price'] is not None and fill['qty'] is not None:
                # For a short, PnL = (entry - exit) * quantity
                total_pnl += (entry_price - float(fill['price'])) * float(fill['qty'])

        await self.db.update_position(
            pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=total_pnl
        )
        await self.risk.on_trade_close(total_pnl, self.tg)
        
        del self.open_positions[pid]
        self.last_exit[symbol] = datetime.now(timezone.utc)
        await self.tg.send(f"✅ {symbol} position closed. Total PnL ≈ {total_pnl:.2f} USDT")

    async def _force_close_position(self, pid: int, pos: Dict[str, Any], tag: str):
        symbol = pos["symbol"]
        try:
            await self.exchange.cancel_all_orders(symbol)
            await self.exchange.create_market_order(symbol, "buy", pos["size"], params={'reduceOnly': True})
        except Exception as e:
            LOG.warning("Force-close order issue on %s: %s", symbol, e)

        last_price = (await self.exchange.fetch_ticker(symbol))["last"]
        pnl = (pos["entry_price"] - last_price) * pos["size"]
        await self.db.update_position(pid, status="CLOSED", closed_at=datetime.now(timezone.utc), pnl=pnl)
        await self.db.add_fill(pid, tag, last_price, pos["size"], datetime.now(timezone.utc))
        await self.risk.on_trade_close(pnl, self.tg)
        self.last_exit[symbol] = datetime.now(timezone.utc)
        del self.open_positions[pid]
        await self.tg.send(f"⏰ {symbol} closed by {tag}. PnL ≈ {pnl:.2f} USDT")

    async def _main_signal_loop(self):
        LOG.info("Starting main signal scan loop.")
        while True:
            try:
                if self.paused or not self.risk.can_trade():
                    await asyncio.sleep(5)
                    continue

                LOG.info("Starting new scan cycle for %d symbols...", len(self.symbols))
                equity = await self.db.latest_equity() or 0.0
                open_positions_count = len(self.open_positions)
                open_symbols = {p['symbol'] for p in self.open_positions.values()}

                for sym in self.symbols:
                    if self.paused or not self.risk.can_trade():
                        break
                    if sym in open_symbols:
                        continue

                    cd_h = self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS)
                    last_x = self.last_exit.get(sym)
                    if last_x and datetime.now(timezone.utc) - last_x < timedelta(hours=cd_h):
                        continue

                    async with self.symbol_locks[sym]:
                        signal = await self._scan_symbol_for_signal(sym)
                        if signal:
                            age = (datetime.utcnow().date() - self._listing_dates_cache[sym]).days if sym in self._listing_dates_cache else None
                            ok, vetoes = filters.evaluate(
                                signal, listing_age_days=age,
                                open_positions=open_positions_count, equity=equity
                            )
                            if ok:
                                await self._open_position(signal)
                                open_positions_count += 1
                                open_symbols.add(sym)
                            else:
                                LOG.info("Signal for %s vetoed: %s", sym, " | ".join(vetoes))
                    await asyncio.sleep(0.5)

            except Exception as e:
                LOG.error("Critical error in main signal loop: %s", e)
                traceback.print_exc()

            LOG.info("Scan cycle complete. Waiting for next interval.")
            await asyncio.sleep(self.cfg.get("SCAN_INTERVAL_SEC", 60))

    async def _equity_loop(self):
        while True:
            try:
                bal = await self._fetch_platform_balance()
                current_equity = bal["total"]["USDT"]
                await self.db.snapshot_equity(current_equity, datetime.now(timezone.utc))

                if current_equity > self.peak_equity:
                    self.peak_equity = current_equity

                if self.cfg.get("DD_PAUSE_ENABLED", True) and not self.risk.kill_switch:
                    if self.peak_equity and current_equity < self.peak_equity:
                        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
                        max_dd_pct = self.cfg.get("DD_MAX_PCT", 10.0)
                        if drawdown_pct >= max_dd_pct:
                            self.risk.kill_switch = True
                            msg = f"❌ KILL-SWITCH: Equity drawdown of {drawdown_pct:.2f}% exceeded {max_dd_pct}%."
                            LOG.warning(msg)
                            await self.tg.send(msg)
            except Exception as e:
                LOG.error("Error in equity loop: %s", e)
            await asyncio.sleep(3600)

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

    async def _resume(self):
        """
        On startup, load state from DB and reconcile with actual
        exchange positions. Includes a definitive "clean slate" protocol that
        correctly fetches all order types using the proper V5 API parameters.
        """
        LOG.info("--> Resuming state...")

        # --- FINAL, CORRECTED: Clean Slate Protocol ---
        LOG.info("Step 1: Fetching and cancelling all old open orders...")
        try:
            # Bybit's V5 API requires fetching different order types separately.
            # We fetch all of them to ensure a truly clean slate.
            
            # 1. Define the parameters for each type of order we need to fetch.
            fetch_params = [
                {'category': 'linear'},                                  # Regular open orders
                {'category': 'linear', 'orderFilter': 'StopOrder'},      # Conditional Stop/Trigger orders
                {'category': 'linear', 'orderFilter': 'tpslOrder'},      # Position-linked TP/SL orders
            ]

            # 2. Fetch all order types concurrently.
            all_orders_lists = await asyncio.gather(
                self.exchange.fetch_open_orders(params=fetch_params[0]),
                self.exchange.fetch_open_orders(params=fetch_params[1]),
                self.exchange.fetch_open_orders(params=fetch_params[2]),
                return_exceptions=True
            )

            # 3. Combine the results into a single list.
            all_open_orders = []
            for result in all_orders_lists:
                if isinstance(result, list):
                    all_open_orders.extend(result)

            if all_open_orders:
                # 4. Group the found orders by their symbol.
                # We do NOT filter by symbols.txt, to ensure we cancel truly orphaned orders.
                orders_by_symbol = defaultdict(list)
                for order in all_open_orders:
                    orders_by_symbol[order['symbol']].append(order['id'])

                if orders_by_symbol:
                    LOG.info("Found %d leftover orders across %d symbols. Cancelling all...", 
                             len(all_open_orders), len(orders_by_symbol))
                    
                    # 5. Loop through each symbol and issue a single cancel_all_orders command.
                    # This is efficient and robust.
                    cancel_tasks = [
                        self.exchange.cancel_all_orders(symbol, params={'category': 'linear'}) 
                        for symbol in orders_by_symbol.keys()
                    ]
                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                    
                    LOG.info("...Successfully cancelled old orders.")
                else:
                    LOG.info("...No old open orders found. Clean slate confirmed.")
            else:
                LOG.info("...No old open orders found. Clean slate confirmed.")

        except Exception as e:
            LOG.error("CRITICAL: Failed to perform clean slate protocol on startup: %s", e)
            traceback.print_exc()

        # --- END OF NEW LOGIC ---

        LOG.info("Step 2: Fetching peak equity from DB...")
        peak = await self.db.pool.fetchval("SELECT MAX(equity) FROM equity_snapshots")
        self.peak_equity = float(peak) if peak is not None else 0.0
        LOG.info("...Initial peak equity loaded: $%.2f", self.peak_equity)

        LOG.info("Step 3: Fetching open positions from the EXCHANGE...")
        try:
            exchange_positions = await self.exchange.fetch_positions()
            open_exchange_positions = {
                p['info']['symbol']: p for p in exchange_positions if float(p['info']['size']) > 0
            }
            LOG.info("...Success! Found %d positions on the exchange.", len(open_exchange_positions))
        except Exception as e:
            LOG.error("Could not fetch exchange positions on startup: %s. Exiting.", e)
            sys.exit(1)

        LOG.info("Step 4: Fetching 'OPEN' positions from the DATABASE...")
        db_positions = {r["symbol"]: dict(r) for r in await self.db.fetch_open_positions()}
        LOG.info("...Success! Found %d 'OPEN' positions in the database.", len(db_positions))
        
        LOG.info("Step 5: Reconciling DB and exchange positions...")
        for symbol, pos_data in open_exchange_positions.items():
            if symbol not in db_positions:
                msg = (
                    f"🚨 ORPHAN DETECTED: Position for {symbol} exists on the exchange "
                    f"but not in the database. Forcing immediate close to manage risk."
                )
                LOG.warning(msg)
                await self.tg.send(msg)
                try:
                    side = 'buy' if pos_data['side'] == 'short' else 'sell'
                    size = float(pos_data['info']['size'])
                    await self.exchange.create_market_order(symbol, side, size, params={'reduceOnly': True})
                except Exception as e:
                    LOG.error("Failed to force-close orphan position %s: %s", symbol, e)

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

        for symbol, db_pos in db_positions.items():
            if symbol in open_exchange_positions:
                ex_pos = open_exchange_positions[symbol]
                db_size = float(db_pos['size'])
                ex_size = float(ex_pos['info']['size'])
                
                if abs(db_size - ex_size) > 1e-9:
                    pid = db_pos["id"]
                    msg = (
                        f"🚨 SIZE MISMATCH on resume for {symbol} (pid {pid}): "
                        f"DB size is {db_size}, Exchange size is {ex_size}. "
                        f"Flagging for manual review."
                    )
                    LOG.critical(msg)
                    await self.tg.send(msg)
                    await self.db.update_position(pid, status="SIZE_MISMATCH")
                    del db_positions[symbol]

        LOG.info("...Reconciliation complete.")

        LOG.info("Step 6: Loading final reconciled positions into memory...")
        final_open_rows = await self.db.fetch_open_positions()
        for r in final_open_rows:
            self.open_positions[r["id"]] = dict(r)
            LOG.info("...Successfully resumed and verified open position for %s (ID: %d)", r["symbol"], r["id"])

        LOG.info("Step 7: Loading recent exit timestamps for cooldowns...")
        cd_h = int(self.cfg.get("SYMBOL_COOLDOWN_HOURS",
                        cfg.SYMBOL_COOLDOWN_HOURS))
        rows = await self.db.pool.fetch(
            "SELECT symbol, closed_at FROM positions "
            "WHERE status='CLOSED' AND closed_at > (NOW() AT TIME ZONE 'utc') - $1::interval",
            timedelta(hours=cd_h),
        )
        for r in rows:
            self.last_exit[r["symbol"]] = r["closed_at"]
        
        LOG.info("<-- Resume complete.")

    async def run(self):
        await self.db.init()
        
        if self.settings.bybit_testnet:
            LOG.warning("="*60)
            LOG.warning("RUNNING ON TESTNET")
            LOG.warning("Testnet data is unreliable for most altcoins.")
            LOG.warning("Signals may be incorrect. Use for order logic testing only.")
            LOG.warning("="*60)

        LOG.info("Loading exchange market data...")
        try:
            await self.exchange._exchange.load_markets()
            LOG.info("Market data loaded.")
        except Exception as e:
            LOG.error("Could not load exchange market data: %s. Exiting.", e)
            return

        LOG.info("Loading symbol listing dates...")
        self._listing_dates_cache = await self._load_listing_dates()

        await self._resume()
        await self.tg.send("🤖 Bot online v3.0")

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
            await self.exchange.close()
            if self.db.pool:
                await self.db.pool.close()
            await self.tg.close()
            LOG.info("Bot shut down cleanly.")


###############################################################################
# 6 ▸ ENTRYPOINT ##############################################################
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