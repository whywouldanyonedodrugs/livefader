"""
live_trader.py – v3.0 (Simplified & Corrected)
===========================================================================
This version restores the missing _load_listing_dates method that was
accidentally deleted, fixing the startup crash.
"""

from __future__ import annotations

import secrets
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
import subprocess
import math
import aiohttp
import asyncpg
import ccxt.async_support as ccxt
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from live.winprob_loader import WinProbScorer


import config as cfg
from . import indicators as ta
from live.indicators import vwap_stack_features
from . import filters
from . shared_utils import is_blacklisted
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .exchange_proxy import ExchangeProxy
from .database import DB
from .telegram import TelegramBot

from pydantic import Field, Extra, ValidationError
from pydantic_settings import BaseSettings

# Shim so joblib can unpickle research ModelBundle saved elsewhere
class ModelBundle:
    pass

# Cyclical hour encoder: returns [sin(hour*2π/24), cos(hour*2π/24)]
def _hour_cyc(X):
    import numpy as np
    import pandas as pd
    h = np.asarray(X).astype(float).reshape(-1, 1)
    sin = np.sin(2 * np.pi * h / 24.0)
    cos = np.cos(2 * np.pi * h / 24.0)
    return np.hstack([sin, cos])

# Day-of-week one-hot encoder: 7 columns, Mon=0..Sun=6 (or whatever you used)
def _dow_onehot(X):
    import numpy as np
    import pandas as pd
    d = np.asarray(X).astype(int).reshape(-1, 1)
    out = np.zeros((d.shape[0], 7), dtype=float)
    m = (d >= 0) & (d < 7)
    out[np.arange(d.shape[0])[m.ravel()], d[m].ravel()] = 1.0
    return out

def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Calculates a Triangular Moving Average (TMA). Helper function."""
    # This is a simplified and faster version of TMA
    return series.rolling(window=period, min_periods=period).mean().rolling(window=period, min_periods=period).mean()

def load_config(path: str | None = None):
    candidates = [
        Path(path) if path else None,
        Path(os.getenv("CONFIG_PATH", "")) if os.getenv("CONFIG_PATH") else None,
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent / "config.yaml",
        Path("/live/config.yaml"),
    ]
    for p in [c for c in candidates if c]:
        if p.exists():
            with open(p, "r") as f:
                cfg = yaml.safe_load(f)  # safe parser recommended. :contentReference[oaicite:2]{index=2}
            print(f"[config] loaded: {p.resolve()}")
            print("[config] sizing snapshot:", {
                "WINPROB_SIZING_ENABLED": cfg.get("WINPROB_SIZING_ENABLED"),
                "WINPROB_PROB_FLOOR": cfg.get("WINPROB_PROB_FLOOR", cfg.get("WINPROB_SIZE_FLOOR")),
                "WINPROB_PROB_CAP":   cfg.get("WINPROB_PROB_CAP",   cfg.get("WINPROB_SIZE_CAP")),
                "WINPROB_MIN_MULTIPLIER": cfg.get("WINPROB_MIN_MULTIPLIER"),
                "WINPROB_MAX_MULTIPLIER": cfg.get("WINPROB_MAX_MULTIPLIER"),
                "VWAP_STACK_SIZING_ENABLED": cfg.get("VWAP_STACK_SIZING_ENABLED"),
                "VWAP_STACK_MIN_MULTIPLIER": cfg.get("VWAP_STACK_MIN_MULTIPLIER"),
                "VWAP_STACK_MAX_MULTIPLIER": cfg.get("VWAP_STACK_MAX_MULTIPLIER"),
            })
            return cfg
    raise FileNotFoundError("config.yaml not found in expected locations")

class RegimeDetector:
    """
    Calculates and caches the market regime based on a live benchmark asset.
    """
    def __init__(self, exchange, config: dict):
        self.exchange = exchange
        self.cfg = config
        self.benchmark_symbol = self.cfg.get("REGIME_BENCHMARK_SYMBOL", "BTCUSDT")
        self.cache_duration = timedelta(minutes=self.cfg.get("REGIME_CACHE_MINUTES", 60))
        self.cached_regime = "UNKNOWN"
        self.last_calculation_time = None
        LOG.info("RegimeDetector initialized for benchmark %s with a %d-minute cache.", self.benchmark_symbol, self.cfg.get("REGIME_CACHE_MINUTES", 60))

    async def _fetch_benchmark_data(self) -> pd.DataFrame | None:
        """Fetches the last ~500 days of daily OHLCV data for the benchmark symbol."""
        try:
            # Fetch enough data for the long-term TMA and Markov model
            ohlcv = await self.exchange.fetch_ohlcv(self.benchmark_symbol, '1d', limit=500)
            if len(ohlcv) < 200: # Need at least ~200 days for meaningful calculation
                LOG.warning("Not enough historical data for benchmark %s to calculate regime. Found %d bars.", self.benchmark_symbol, len(ohlcv))
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            LOG.error("Failed to fetch benchmark data for regime detection: %s", e)
            return None

    def _calculate_vol_regime(self, daily_returns: pd.Series) -> pd.Series:
        """Calculates the volatility regime using a Markov Switching Model."""
        try:
            model = sm.tsa.MarkovRegression(daily_returns.dropna(), k_regimes=2, switching_variance=True)
            results = model.fit(disp=False)
            low_vol_regime_idx = np.argmin(results.params[-2:])
            vol_regimes = np.where(results.smoothed_marginal_probabilities[low_vol_regime_idx] > 0.5, "LOW_VOL", "HIGH_VOL")
            # Return as a pandas Series with the same index as the input
            return pd.Series(vol_regimes, index=daily_returns.dropna().index, name="vol_regime")
        except Exception as e:
            LOG.warning("Could not fit Markov model for volatility regime: %s. Defaulting to UNKNOWN.", e)
            return pd.Series("UNKNOWN", index=daily_returns.index, name="vol_regime")

    def _calculate_trend_regime(self, df_daily: pd.DataFrame) -> pd.Series:
        """Calculates the trend regime using a TMA and Keltner Channel."""
        df_daily['tma'] = triangular_moving_average(df_daily['close'], self.cfg.get("REGIME_MA_PERIOD", 100))
        atr_series = ta.atr(df_daily, period=self.cfg.get("REGIME_ATR_PERIOD", 20))
        df_daily['keltner_upper'] = df_daily['tma'] + (atr_series * self.cfg.get("REGIME_ATR_MULT", 2.0))
        df_daily['keltner_lower'] = df_daily['tma'] - (atr_series * self.cfg.get("REGIME_ATR_MULT", 2.0))
        df_daily.dropna(inplace=True)

        trend = pd.Series(np.nan, index=df_daily.index, dtype="object")
        for i in range(1, len(df_daily)):
            if df_daily['close'].iloc[i] > df_daily['keltner_upper'].iloc[i]:
                trend.iloc[i] = "BULL"
            elif df_daily['close'].iloc[i] < df_daily['keltner_lower'].iloc[i]:
                trend.iloc[i] = "BEAR"
            else:
                trend.iloc[i] = trend.iloc[i-1] # Carry forward the previous trend
        
        return trend.ffill().bfill()

    async def get_current_regime(self) -> str:
        """The main public method to get the current market regime, using a cache."""
        now = datetime.now(timezone.utc)
        if self.last_calculation_time and (now - self.last_calculation_time) < self.cache_duration:
            return self.cached_regime

        LOG.info("Regime cache expired or empty. Calculating new market regime...")
        self.last_calculation_time = now # Update time immediately to prevent re-entry

        df_daily = await self._fetch_benchmark_data()
        if df_daily is None:
            self.cached_regime = "UNKNOWN"
            return self.cached_regime

        daily_returns = df_daily['close'].pct_change()
        
        vol_regime = self._calculate_vol_regime(daily_returns)
        df_daily['vol_regime'] = vol_regime
        
        trend_regime = self._calculate_trend_regime(df_daily)
        df_daily['trend_regime'] = trend_regime
        
        df_daily.dropna(subset=['vol_regime', 'trend_regime'], inplace=True)
        
        if df_daily.empty:
            LOG.warning("Regime calculation resulted in an empty DataFrame. Cannot determine regime.")
            self.cached_regime = "UNKNOWN"
            return self.cached_regime

        latest_regime = f"{df_daily['trend_regime'].iloc[-1]}_{df_daily['vol_regime'].iloc[-1]}"
        self.cached_regime = latest_regime
        
        LOG.info("New market regime calculated: %s", self.cached_regime)
        return self.cached_regime

def create_unique_cid(tag: str) -> str:
    """Creates a globally unique client order ID for one-shot orders like entry."""
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # The random hex makes it unique even if called in the same millisecond.
    random_suffix = secrets.token_hex(2)
    return f"bot_{tag}_{timestamp_ms}_{random_suffix}"[:36]

def create_stable_cid(pid: int, tag: str) -> str:
    """Creates a stable, repeatable client order ID for manageable orders like SL/TP."""
    return f"bot_{pid}_{tag}"[:36]

@dataclasses.dataclass
class Signal:
    # --- All required fields (no default value) ---
    symbol: str
    entry: float
    atr: float
    rsi: float
    adx: float
    atr_pct: float
    market_regime: str
    price_boom_pct: float
    price_slowdown_pct: float
    vwap_dev_pct: float
    ret_30d: float
    ema_fast: float
    ema_slow: float
    listing_age_days: int
    vwap_z_score: float
    is_ema_crossed_down: bool
    vwap_consolidated: bool
    session_tag: str
    day_of_week: int
    hour_of_day: int

    # --- All optional fields (with a default value) ---
    win_probability: float = 0.0

    def __post_init__(self):
        """This function runs after the object is created."""
        LOG.info(f"Successfully created Signal object for {self.symbol} using correct class definition.")




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
        
        # FIX: Create self.exchange FIRST...
        self.exchange = ExchangeProxy(self._init_ccxt())
        
        # FIX: ...THEN create self.regime_detector, which depends on it.
        self.regime_detector = RegimeDetector(self.exchange, self.cfg)

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

        self.winprob = WinProbScorer()
        if self.winprob.is_loaded:
            LOG.info("WinProb ready (kind=%s, features=%d)", self.winprob.kind, len(self.winprob.expected_features))
        else:
            LOG.warning("WinProb not loaded; using wp=0.0")        

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

    def _normalize_cfg_key(self, key: str) -> str:
        """Alias old → new config keys so /set works with either."""
        key_up = key.upper()
        aliases = {
            # risk
            "FIXED_RISK_USDT": "RISK_USD",
            "RISK_USDT": "RISK_USD",
            "RISK_PCT": "RISK_EQUITY_PCT",   # keep old name working
            # winprob (old prob bounds)
            "WINPROB_SIZE_FLOOR": "WINPROB_PROB_FLOOR",
            "WINPROB_SIZE_CAP":   "WINPROB_PROB_CAP",
        }
        return aliases.get(key_up, key_up)

    @staticmethod
    def _cid(pid: int, tag: str) -> str:
        """Creates a unique, valid client order ID."""
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Bybit V5 requires clientOrderId to be 36 chars or less.
        return f"bot_{pid}_{tag}_{timestamp_ms}"[:36]

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

    async def _load_listing_dates(self) -> Dict[str, datetime.date]:
        if LISTING_PATH.exists():
            raw = json.loads(LISTING_PATH.read_text())
            return {s: datetime.fromisoformat(ts).date() for s, ts in raw.items()}

        LOG.info("listing_dates.json not found. Fetching from exchange.")
        async def fetch_date(sym):
            try:
                candles = await self.exchange.fetch_ohlcv(sym, timeframe="1d", limit=1, since=0)
                if candles:
                    ts = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
                    return sym, ts.date()
            except Exception as e:
                LOG.warning("Could not fetch listing date for %s: %s", sym, e)
            return sym, None

        results = await asyncio.gather(*(fetch_date(s) for s in self.symbols))
        out = {sym: d for sym, d in results if d}
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

    async def _risk_amount(self, free_usdt: float, eth_macd_data: Optional[dict]) -> float:
        """
        Returns the USD risk budget for THIS trade after applying ETH barometer rules.
        - Base risk from config (percent of free USDT or fixed).
        - If ETH barometer is unfavorable (hist > 0 for shorts), either BLOCK or RESIZE.
        """
        mode = str(self.cfg.get("RISK_MODE", "PERCENT")).upper()
        base_risk = (free_usdt * float(self.cfg["RISK_PCT"])) if mode == "PERCENT" else float(self.cfg["FIXED_RISK_USDT"])

        # Barometer is optional and safe by default
        if self.cfg.get("ETH_BAROMETER_ENABLED", False) and eth_macd_data is not None:
            # For a SHORT-only system, "unfavorable" = ETH momentum up (hist > 0)
            unfavorable = float(eth_macd_data.get("hist", 0)) > 0
            if unfavorable:
                mode = str(self.cfg.get("UNFAVORABLE_MODE", "BLOCK")).upper()
                if mode == "BLOCK":
                    LOG.info("ETH barometer UNFAVORABLE → veto entry (risk=0).")
                    return 0.0
                if mode == "RESIZE":
                    f = float(self.cfg.get("UNFAVORABLE_RISK_RESIZE_FACTOR", 0.5))
                    resized = base_risk * f
                    LOG.info(f"ETH barometer UNFAVORABLE → resize risk {base_risk:.2f} → {resized:.2f} (x{f}).")
                    return resized

        LOG.info(f"ETH barometer favorable/disabled → using base risk {base_risk:.2f}.")
        return base_risk


    # --- NEW, CORRECTED HELPER METHODS ---
    async def _fetch_by_cid(self, cid: str, symbol: str):
        """Fetches an order using the robust generic fetch_order method."""
        return await self.exchange.fetch_order(
            None, symbol, params={"clientOrderId": cid, "category": "linear"}
        )

    async def _cancel_by_cid(self, cid: str, symbol: str):
        """Cancels an order using the robust generic cancel_order method."""
        try:
            return await self.exchange.cancel_order(
                None, symbol, params={"clientOrderId": cid, "category": "linear"}
            )
        except ccxt.OrderNotFound:
            LOG.warning("Order %s for %s already filled or cancelled, no action needed.", cid, symbol)
        except Exception as e:
            LOG.error("Failed to cancel order %s for %s: %s", cid, symbol, e)

    async def _all_open_orders(self, symbol: str) -> list:
        """
        Fetches all types of open orders (regular, conditional, TP/SL) for a specific symbol
        by correctly using the Bybit V5 API parameters.
        """
        params_linear = {'category': 'linear'}
        try:
            # Fetch orders for the specific symbol passed as an argument
            active_orders = await self.exchange.fetch_open_orders(symbol, params=params_linear)
            stop_orders = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'StopOrder'})
            tpsl_orders = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'tpslOrder'})
            return active_orders + stop_orders + tpsl_orders
        except Exception as e:
            LOG.warning("Could not fetch all open orders for %s: %s", symbol, e)
            return [] # Return an empty list on failure to prevent crashes

    async def _all_open_orders_for_all_symbols(self) -> list:
        """
        Fetches all types of open orders (regular, conditional, TP/SL) for all symbols.
        """
        params_linear = {'category': 'linear'}
        try:
            active_orders = await self.exchange.fetch_open_orders(params=params_linear)
            stop_orders = await self.exchange.fetch_open_orders(params={**params_linear, 'orderFilter': 'StopOrder'})
            tpsl_orders = await self.exchange.fetch_open_orders(params={**params_linear, 'orderFilter': 'tpslOrder'})
            return active_orders + stop_orders + tpsl_orders
        except Exception as e:
            LOG.warning("Could not fetch all open orders for all symbols: %s", e)
            return []

    async def _scan_symbol_for_signal(self, symbol: str, market_regime: str, eth_macd: Optional[dict]) -> Optional[Signal]:
        LOG.info("Checking %s...", symbol)
        try:
            # ---- Timeframes & OHLCV fetch ----
            base_tf = self.cfg.get('TIMEFRAME', '5m')
            ema_tf  = self.cfg.get('EMA_TIMEFRAME', '4h')
            rsi_tf  = self.cfg.get('RSI_TIMEFRAME', '1h')
            atr_tf  = self.cfg.get('ADX_TIMEFRAME', '1h')
            required_tfs = {base_tf, ema_tf, rsi_tf, atr_tf, '1d'}

            async with self.api_semaphore:
                tasks = {tf: self.exchange.fetch_ohlcv(symbol, tf, limit=500) for tf in required_tfs}
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                ohlcv_data = dict(zip(tasks.keys(), results))

            dfs: dict[str, pd.DataFrame] = {}
            for tf, data in ohlcv_data.items():
                if isinstance(data, Exception) or not data:
                    LOG.debug("Could not fetch OHLCV for %s on %s timeframe.", symbol, tf)
                    return None
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)

                # Stale/illiquid check on the most recent 100 completed bars
                recent_candles = df.tail(100)
                if not recent_candles.empty:
                    zero_volume_pct = (recent_candles['volume'] == 0).sum() / len(recent_candles)
                    if zero_volume_pct > 0.25:
                        LOG.warning("DATA_ERROR for %s on %s: Detected stale data (%.0f%% zero volume). Skipping.",
                                    symbol, tf, zero_volume_pct * 100)
                        return None

                # Drop the last (possibly incomplete) bar
                if len(df) < 3:
                    return None
                df = df.iloc[:-1]
                if df.empty:
                    return None
                dfs[tf] = df

            df5 = dfs[base_tf]

            # ---- Indicators aligned to base_tf index ----
            df5['ema_fast'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_FAST_PERIOD).reindex(df5.index, method='ffill')
            df5['ema_slow'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_SLOW_PERIOD).reindex(df5.index, method='ffill')
            df5['rsi']      = ta.rsi(dfs[rsi_tf]['close'], cfg.RSI_PERIOD).reindex(df5.index, method='ffill')
            df5['atr']      = ta.atr(dfs[atr_tf], cfg.ADX_PERIOD).reindex(df5.index, method='ffill')
            df5['adx']      = ta.adx(dfs[atr_tf], cfg.ADX_PERIOD).reindex(df5.index, method='ffill')

            # ---- VWAP-stack features (safe; no signal_obj here) ----
            lookback = int(self.cfg.get("VWAP_STACK_LOOKBACK_BARS", 12))
            band_pct = float(self.cfg.get("VWAP_STACK_BAND_PCT", 0.004))
            try:
                vw = vwap_stack_features(df5[['open','high','low','close','volume']].copy(),
                                        lookback_bars=lookback, band_pct=band_pct)
                vwap_frac  = float(vw.get("vwap_frac_in_band", 0.0))
                vwap_exp   = float(vw.get("vwap_expansion_pct", 0.0))
                vwap_slope = float(vw.get("vwap_slope_pph", 0.0))
            except Exception as e:
                LOG.error("VWAP-stack calc failed for %s: %s", symbol, e)
                vwap_frac = vwap_exp = vwap_slope = 0.0

            # ---- Boom/slowdown & GAP VWAP (existing logic) ----
            tf_minutes = 5  # base_tf is '5m' in your config
            boom_bars = int((cfg.PRICE_BOOM_PERIOD_H * 60) / tf_minutes)
            slowdown_bars = int((cfg.PRICE_SLOWDOWN_PERIOD_H * 60) / tf_minutes)
            df5['price_boom_ago'] = df5['close'].shift(boom_bars)
            df5['price_slowdown_ago'] = df5['close'].shift(slowdown_bars)

            df1d = dfs['1d']
            ret_30d = (df1d['close'].iloc[-1] / df1d['close'].iloc[-cfg.STRUCTURAL_TREND_DAYS] - 1) \
                    if len(df1d) > cfg.STRUCTURAL_TREND_DAYS else 0.0

            vwap_bars = int((cfg.GAP_VWAP_HOURS * 60) / tf_minutes)
            vwap_num = (df5['close'] * df5['volume']).shift(1).rolling(vwap_bars).sum()
            vwap_den = df5['volume'].shift(1).rolling(vwap_bars).sum()
            df5['vwap'] = vwap_num / vwap_den
            vwap_dev_raw = df5['close'] - df5['vwap']
            df5['vwap_dev_pct'] = vwap_dev_raw / df5['vwap']
            df5['price_std'] = df5['close'].rolling(vwap_bars).std()
            df5['vwap_z_score'] = vwap_dev_raw / df5['price_std']
            df5['vwap_ok'] = df5['vwap_dev_pct'].abs() <= cfg.GAP_MAX_DEV_PCT
            df5['vwap_consolidated'] = df5['vwap_ok'].rolling(cfg.GAP_MIN_BARS).min().fillna(0).astype(bool)

            df5.dropna(inplace=True)
            if df5.empty:
                return None

            last = df5.iloc[-1]
            is_ema_crossed_down = last['ema_fast'] < last['ema_slow']
            now_utc = datetime.now(timezone.utc)
            boom_ret_pct = (last['close'] / last['price_boom_ago'] - 1)
            slowdown_ret_pct = (last['close'] / last['price_slowdown_ago'] - 1)
            price_boom = boom_ret_pct > cfg.PRICE_BOOM_PCT
            price_slowdown = slowdown_ret_pct < cfg.PRICE_SLOWDOWN_PCT
            atr_pct = (last['atr'] / last['close']) * 100 if last['close'] > 0 else 0.0
            listing_age_days = (now_utc.date() - self._listing_dates_cache[symbol]).days if symbol in self._listing_dates_cache else -1
            hour_of_day = now_utc.hour
            day_of_week = now_utc.weekday()
            session_tag = "ASIA" if 0 <= hour_of_day < 8 else "EUROPE" if 8 <= hour_of_day < 16 else "US"

            ema_down = True
            ema_log_msg = " DISABLED"
            if self.cfg.get("EMA_FILTER_ENABLED", True):
                ema_down = is_ema_crossed_down
                ema_log_msg = f"{'✅' if ema_down else '❌'} (Fast: {last['ema_fast']:.4f} < Slow: {last['ema_slow']:.4f})"

            trend_log_msg = " DISABLED"
            if self.cfg.get("STRUCTURAL_TREND_FILTER_ENABLED", True):
                trend_ok = ret_30d <= self.cfg.get("STRUCTURAL_TREND_RET_PCT", 0.01)
                trend_log_msg = f"{'✅' if trend_ok else '❌'} (Return: {ret_30d:+.2%})"

            vwap_log_msg = " DISABLED"
            if self.cfg.get("GAP_FILTER_ENABLED", True):
                vwap_dev_pct = last.get('vwap_dev_pct', float('nan'))
                vwap_dev_str = f"{vwap_dev_pct:.2%}" if pd.notna(vwap_dev_pct) else "N/A"
                current_dev_ok = last.get('vwap_ok', False)
                vwap_log_msg = f"{'✅' if last['vwap_consolidated'] else '❌'} (Streak Failed) | Current Dev: {'✅' if current_dev_ok else '❌'} ({vwap_dev_str})"

            LOG.debug(
                f"\n--- {symbol} | {last.name.strftime('%Y-%m-%d %H:%M')} UTC ---\n"
                f"  [Base Timeframe: {base_tf}]\n"
                f"  - Price Boom     (>{cfg.PRICE_BOOM_PCT:.0%}, {cfg.PRICE_BOOM_PERIOD_H}h): {'✅' if price_boom else '❌'} ({boom_ret_pct:+.2%})\n"
                f"  - Price Slowdown (<{cfg.PRICE_SLOWDOWN_PCT:.0%}, {cfg.PRICE_SLOWDOWN_PERIOD_H}h): {'✅' if price_slowdown else '❌'} ({slowdown_ret_pct:+.2%})\n"
                f"  - EMA Trend Down ({ema_tf}):      {ema_log_msg}\n"
                f"  --------------------------------------------------\n"
                f"  - RSI ({rsi_tf}):                 {last['rsi']:.2f} (Veto: {not (cfg.RSI_ENTRY_MIN <= last['rsi'] <= cfg.RSI_ENTRY_MAX)})\n"
                f"  - 30d Trend Filter:        {trend_log_msg}\n"
                f"  - VWAP Consolidated:       {vwap_log_msg}\n"
                f"  - ATR ({atr_tf}):                 {last['atr']:.6f}\n"
                f"  - VWAP stack: frac={vwap_frac:.2f}, exp={vwap_exp*100:.2f}%, slope_pph={vwap_slope:.4f}\n"
                f"====================================================\n"
            )

            # ---- Entry gate ----
            if price_boom and price_slowdown and ema_down:
                LOG.info("SIGNAL FOUND for %s at price %.4f", symbol, last['close'])

                signal_obj = Signal(
                    symbol=symbol, entry=float(last['close']), atr=float(last['atr']),
                    rsi=float(last['rsi']), adx=float(last['adx']), atr_pct=atr_pct,
                    market_regime=market_regime, price_boom_pct=boom_ret_pct,
                    price_slowdown_pct=slowdown_ret_pct, vwap_dev_pct=float(last.get('vwap_dev_pct', 0.0)),
                    vwap_z_score=float(last.get('vwap_z_score', 0.0)), ret_30d=ret_30d,
                    ema_fast=float(last['ema_fast']), ema_slow=float(last['ema_slow']),
                    listing_age_days=listing_age_days, session_tag=session_tag,
                    day_of_week=day_of_week, hour_of_day=hour_of_day,
                    vwap_consolidated=bool(last.get('vwap_consolidated', False)),
                    is_ema_crossed_down=bool(is_ema_crossed_down)
                )

                # attach VWAP-stack diagnostics to the signal (used later for sizing + DB)
                signal_obj.vwap_stack_frac = vwap_frac
                signal_obj.vwap_stack_expansion_pct = vwap_exp
                signal_obj.vwap_stack_slope_pph = vwap_slope

                # ---- Win-prob scoring (statsmodels or sklearn via adapter) ----
                try:
                    eth_bar = await self._get_eth_macd_barometer()
                    eth_hist = float(eth_bar.get("hist", 0.0)) if eth_bar else 0.0

                    # Build the live feature dict once
                    live_data = {
                        "rsi_at_entry": float(signal_obj.rsi),
                        "adx_at_entry": float(signal_obj.adx),
                        "price_boom_pct_at_entry": float(signal_obj.price_boom_pct),
                        "price_slowdown_pct_at_entry": float(signal_obj.price_slowdown_pct),
                        "vwap_z_at_entry": float(signal_obj.vwap_z_score),
                        "ema_spread_pct_at_entry": (
                            (float(signal_obj.ema_fast) - float(signal_obj.ema_slow)) / float(signal_obj.ema_slow)
                            if float(signal_obj.ema_slow) > 0.0 else 0.0
                        ),
                        "is_ema_crossed_down_at_entry": int(bool(signal_obj.is_ema_crossed_down)),
                        "day_of_week_at_entry": int(signal_obj.day_of_week),
                        "hour_of_day_at_entry": int(signal_obj.hour_of_day),
                        "eth_macdhist_at_entry": eth_hist,

                        # Include VWAP-stack features if present in your research model
                        "vwap_stack_frac_at_entry": (
                            float(signal_obj.vwap_stack_frac) if getattr(signal_obj, "vwap_stack_frac", None) is not None else 0.0
                        ),
                        "vwap_stack_expansion_pct_at_entry": (
                            float(signal_obj.vwap_stack_expansion_pct)
                            if getattr(signal_obj, "vwap_stack_expansion_pct", None) is not None else 0.0
                        ),
                        "vwap_stack_slope_pph_at_entry": (
                            float(signal_obj.vwap_stack_slope_pph)
                            if getattr(signal_obj, "vwap_stack_slope_pph", None) is not None else 0.0
                        ),
                    }

                    if hasattr(self, "winprob") and self.winprob.is_loaded:
                        win_prob = self.winprob.score(live_data)  # uses predict_proba for sklearn; adds const for statsmodels internally
                        signal_obj.win_probability = max(0.0, min(1.0, float(win_prob)))
                    else:
                        signal_obj.win_probability = 0.0

                except Exception as e:
                    LOG.warning("Failed to score signal for %s: %s", symbol, e)
                    signal_obj.win_probability = 0.0


                return signal_obj

            return None

        except ccxt.BadSymbol:
            LOG.warning("Could not scan %s: Invalid symbol on exchange.", symbol)
        except Exception as e:
            LOG.error("Error scanning symbol %s: %s", symbol, e)
            traceback.print_exc()
        return None

    # inside live_trader.py (class LiveTrader)

    def _vwap_stack_multiplier(self, stack_frac: float | None, expansion_pct: float | None) -> float:
        """
        VWAP stack multiplier fully controlled by YAML.
        If VWAP_STACK_SIZING_ENABLED is False, returns 1.0.
        If min==max, returns that constant (neutralization).
        Uses your weights to blend 'frac' and 'expansion' → score → multiplier.
        """
        cfg = self.cfg
        if not bool(cfg.get("VWAP_STACK_SIZING_ENABLED", True)):
            return 1.0

        # Read bounds; min==max => neutralize
        m_lo = float(cfg.get("VWAP_STACK_MIN_MULTIPLIER", 1.00))
        m_hi = float(cfg.get("VWAP_STACK_MAX_MULTIPLIER", 1.00))
        if abs(m_hi - m_lo) < 1e-12:
            return m_lo

        # Read thresholds and weights
        frac_min = float(cfg.get("VWAP_STACK_FRAC_MIN", 0.50))
        frac_good = float(cfg.get("VWAP_STACK_FRAC_GOOD", 0.70))
        exp_abs_min = float(cfg.get("VWAP_STACK_EXPANSION_ABS_MIN", 0.006))
        exp_good = float(cfg.get("VWAP_STACK_EXPANSION_GOOD", 0.015))
        w_frac = float(cfg.get("VWAP_STACK_FRAC_WEIGHT", 0.6))
        w_exp  = float(cfg.get("VWAP_STACK_EXP_WEIGHT", 0.4))
        if (w_frac + w_exp) <= 0:
            w_frac = 1.0; w_exp = 0.0

        # Safe inputs
        frac = float(stack_frac) if stack_frac is not None and np.isfinite(stack_frac) else 0.0
        exp  = float(expansion_pct) if expansion_pct is not None and np.isfinite(expansion_pct) else 0.0

        # Normalize each component to [0,1] over its min..good window
        def lin01(x, lo, hi):
            if hi <= lo:
                hi = lo + 1e-6
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        s_frac = lin01(frac, frac_min, frac_good)
        s_exp  = lin01(abs(exp), exp_abs_min, exp_good)  # absolute expansion

        score = (w_frac * s_frac + w_exp * s_exp) / (w_frac + w_exp)

        # Map score [0,1] → [m_lo,m_hi]
        return float(m_lo + score * (m_hi - m_lo))

    async def _get_eth_macd_barometer(self) -> Optional[dict]:
        """
        Returns latest ETHUSDT 4h MACD dict: {'macd': x, 'signal': y, 'hist': z}.
        Used ONLY for gating/resizing risk before order placement.
        """
        try:
            eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=200)
            if not eth_ohlcv:
                return None
            df = pd.DataFrame(eth_ohlcv, columns=['ts','open','high','low','close','volume'])
            macd_df = ta.macd(df['close'])
            latest = macd_df.iloc[-1]
            return {"macd": float(latest['macd']),
                    "signal": float(latest['signal']),
                    "hist": float(latest['hist'])}
        except Exception as e:
            LOG.warning(f"ETH barometer unavailable: {e}")
            return None

    def _winprob_multiplier(self, wp: float) -> float:
        """
        Map a win-probability 'wp' to a size multiplier, controlled by YAML.
        Features:
        - WINPROB_SIZING_ENABLED flag
        - Separate input prob window vs. output multiplier range
        - Backward compatible with WINPROB_SIZE_FLOOR/CAP (as prob bounds)
        - Neutralization when min==max (returns that constant, typically 1.0)
        """
        cfg = self.cfg

        # Feature flag
        if not bool(cfg.get("WINPROB_SIZING_ENABLED", True)):
            return 1.0

        # No probability? No effect.
        try:
            wp = float(wp)
        except Exception:
            return 1.0
        if not np.isfinite(wp) or wp <= 0.0:
            return 1.0
        # Clamp to [0,1] just in case
        wp = float(np.clip(wp, 0.0, 1.0))

        # ---- INPUT PROB WINDOW (domain) ----
        # New names (preferred):
        prob_lo = float(cfg.get("WINPROB_PROB_FLOOR",
                    cfg.get("WINPROB_SIZE_FLOOR", 0.50)))  # back-compat
        prob_hi = float(cfg.get("WINPROB_PROB_CAP",
                    cfg.get("WINPROB_SIZE_CAP",   0.90)))  # back-compat
        if prob_hi <= prob_lo:
            prob_hi = prob_lo + 1e-6

        # ---- OUTPUT MULTIPLIER RANGE (codomain) ----
        # New names (preferred). Defaults replicate your legacy 0.7..1.3 behavior.
        mult_lo = float(cfg.get("WINPROB_MIN_MULTIPLIER", 0.70))
        mult_hi = float(cfg.get("WINPROB_MAX_MULTIPLIER", 1.30))

        # If user set both to the same (e.g., 1.0), neutralize
        if abs(mult_hi - mult_lo) < 1e-12:
            return mult_lo

        # Linear map wp∈[prob_lo,prob_hi] -> [mult_lo,mult_hi], clamped to ends
        x = (np.clip(wp, prob_lo, prob_hi) - prob_lo) / (prob_hi - prob_lo)
        return float(mult_lo + x * (mult_hi - mult_lo))

    async def _open_position(self, sig: Signal) -> None:
        """
        Robust short entry:
        1) Pre-flight checks + risk sizing (ETH barometer, VWAP-stack, win-prob)
        2) Market SELL
        3) Confirm position on exchange
        4) Persist to DB (pid)
        5) Place protective SL + TP(s) with stable clientOrderId
        6) Telegram notify

        If any critical step fails after entry, attempt emergency flatten.
        """
        # --- Pre-flight: block duplicates and exchange-held positions ---
        if any(row["symbol"] == sig.symbol for row in self.open_positions.values()):
            return
        try:
            positions = await self.exchange.fetch_positions(symbols=[sig.symbol])
            if positions and positions[0] and float(positions[0].get("info", {}).get("size", 0)) > 0:
                LOG.warning("Pre-flight found existing exchange position for %s. Abort new entry.", sig.symbol)
                return
        except Exception as e:
            LOG.error("Pre-flight check failed for %s: %s", sig.symbol, e)
            return

        # --- Base risk selection (fixed | percent) ---
        mode = str(self.cfg.get("RISK_MODE", "fixed")).lower()
        fixed_risk = float(self.cfg.get("RISK_USD", 10.0))
        base_risk_usd = fixed_risk
        if mode == "percent":
            try:
                latest_eq = await self.db.latest_equity()
                latest_eq = float(latest_eq) if latest_eq is not None else 0.0
            except Exception:
                latest_eq = 0.0
            pct = float(self.cfg.get("RISK_EQUITY_PCT", 0.01))  # 1% default if used
            base_risk_usd = max(0.0, latest_eq * pct) if latest_eq > 0 else fixed_risk

        # --- Multipliers (environmental & model) ---
        # 1) ETH barometer (inline – async safe)
        eth_mult = 1.0
        if bool(self.cfg.get("ETH_BAROMETER_ENABLED", True)):
            try:
                eth = await self._get_eth_macd_barometer()
                hist = float(eth.get("hist", 0.0)) if eth else 0.0
                cutoff = float(self.cfg.get("ETH_MACD_HIST_CUTOFF_POS", 0.0))
                if hist > cutoff:  # unfavorable for shorts
                    eth_mult = float(self.cfg.get("UNFAVORABLE_RISK_RESIZE_FACTOR", 0.2))
                    LOG.info(
                        "ETH barometer unfavorable → reducing size by %.2f× (hist=%.3f > cutoff=%.3f).",
                        eth_mult, hist, cutoff
                    )
                else:
                    LOG.info("ETH barometer favorable/disabled → using base risk %.2f.", base_risk_usd)
            except Exception as e:
                LOG.warning("ETH barometer unavailable (%s). Proceeding with base risk.", e)

        # 2) VWAP stack multiplier
        vw_mult = self._vwap_stack_multiplier(
            getattr(sig, "vwap_stack_frac", None),
            getattr(sig, "vwap_stack_expansion_pct", None),
        )

        # 3) Calibrated win-prob multiplier
        wp = float(getattr(sig, "win_probability", 0.0) or 0.0)
        wp_mult = self._winprob_multiplier(wp)

        # 4) Combine
        risk_usd = base_risk_usd * eth_mult * vw_mult * wp_mult

        # 5) Optional global clamps from YAML
        if self.cfg.get("RISK_USD_MIN") is not None:
            risk_usd = max(float(self.cfg["RISK_USD_MIN"]), risk_usd)
        if self.cfg.get("RISK_USD_MAX") is not None:
            risk_usd = min(float(self.cfg["RISK_USD_MAX"]), risk_usd)

        sig.risk_usd = float(risk_usd)  

        LOG.info(
            "Sizing → base=%.2f, eth×=%.2f, vwap×=%.2f, wp=%.2f (wp×=%.2f) → risk_usd=%.2f",
            base_risk_usd, eth_mult, vw_mult, wp, wp_mult, risk_usd,
        )

        # --- Compute size from live price / ATR distance ---
        try:
            ticker = await self.exchange.fetch_ticker(sig.symbol)
            px = float(ticker["last"])
        except Exception as e:
            LOG.error("Failed to fetch live ticker for %s: %s", sig.symbol, e)
            return

        sl_mult = float(self.cfg.get("SL_ATR_MULT", 1.8))
        stop_price_preview = px + sl_mult * float(sig.atr)   # short → SL above entry
        stop_dist = abs(px - stop_price_preview)
        if stop_dist <= 0:
            LOG.warning("Stop distance is zero for %s. Skip.", sig.symbol)
            return
        intended_size = max(risk_usd / stop_dist, 0.0)

        # --- Ensure leverage/mode ---
        try:
            await self._ensure_leverage(sig.symbol)
        except Exception:
            return

        # --- Entry: market SELL with unique CID ---
        entry_cid = create_unique_cid(f"ENTRY_{sig.symbol}")
        try:
            await self.exchange.create_market_order(
                sig.symbol, "sell", intended_size,
                params={"clientOrderId": entry_cid, "category": "linear"}
            )
            LOG.info("Market SELL sent for %s (CID=%s)", sig.symbol, entry_cid)
        except Exception as e:
            LOG.error("Market SELL failed for %s: %s", sig.symbol, e)
            return

        # --- Confirm the position actually exists ---
        actual_size, actual_entry_price = 0.0, 0.0
        live_position = None
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                positions = await self.exchange.fetch_positions(symbols=[sig.symbol])
                pos = next((p for p in positions if p.get("info", {}).get("symbol") == sig.symbol), None)
                if pos and float(pos.get("info", {}).get("size", 0)) > 0:
                    live_position = pos
                    actual_size = float(pos["info"]["size"])
                    actual_entry_price = float(pos["info"]["avgPrice"])
                    break
            except Exception as e:
                LOG.warning("Confirm loop failed for %s: %s", sig.symbol, e)

        if not live_position:
            LOG.error("Entry failed to confirm for %s; no exchange position appeared.", sig.symbol)
            return

        # Use the price we just saw pre-entry for slippage vs the actual fill
        slippage_usd = (px - actual_entry_price) * actual_size
        LOG.info("Entry confirmed %s: size=%.6f @ %.6f (slip $%.4f)",
                sig.symbol, actual_size, actual_entry_price, slippage_usd)

        # --- Compute protective levels from *actual* entry ---
        stop_price = actual_entry_price + sl_mult * float(sig.atr)

        # --- Persist position (minimal payload first; then add order CIDs after we create them) ---
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)

        exit_deadline = None
        if self.cfg.get("TIME_EXIT_HOURS_ENABLED", False):
            exit_deadline = now + timedelta(hours=int(self.cfg.get("TIME_EXIT_HOURS", 4)))
        elif self.cfg.get("TIME_EXIT_ENABLED", False):
            exit_deadline = now + timedelta(days=int(self.cfg.get("TIME_EXIT_DAYS", 10)))

        payload = {
            "symbol": sig.symbol,
            "side": "SHORT",
            "size": float(actual_size),
            "entry_price": float(actual_entry_price),
            "trailing_active": False,
            "atr": float(sig.atr),
            "status": "OPEN",
            "opened_at": now,
            "exit_deadline": exit_deadline,

            "entry_cid": entry_cid,

            "market_regime_at_entry": sig.market_regime,
            "risk_usd": float(sig.risk_usd),
            "slippage_usd": float(slippage_usd),

            "rsi_at_entry": float(sig.rsi),
            "adx_at_entry": float(sig.adx),
            "atr_pct_at_entry": float(sig.atr_pct),
            "price_boom_pct_at_entry": float(sig.price_boom_pct),
            "price_slowdown_pct_at_entry": float(sig.price_slowdown_pct),
            "vwap_dev_pct_at_entry": float(getattr(sig, "vwap_dev_pct", 0.0)),
            "vwap_z_at_entry": float(getattr(sig, "vwap_z_score", 0.0)),
            "ret_30d_at_entry": float(sig.ret_30d),
            "ema_fast_at_entry": float(sig.ema_fast),
            "ema_slow_at_entry": float(sig.ema_slow),
            "listing_age_days_at_entry": int(sig.listing_age_days),
            "session_tag_at_entry": sig.session_tag,
            "day_of_week_at_entry": int(sig.day_of_week),
            "hour_of_day_at_entry": int(sig.hour_of_day),
            "vwap_consolidated_at_entry": bool(sig.vwap_consolidated),
            "is_ema_crossed_down_at_entry": bool(sig.is_ema_crossed_down),
            "win_probability_at_entry": float(getattr(sig, "win_probability", 0.0)),

            # VWAP-stack audit (read from sig; they may be None)
            "vwap_stack_frac_at_entry": float(getattr(sig, "vwap_stack_frac", 0.0)) if getattr(sig, "vwap_stack_frac", None) is not None else None,
            "vwap_stack_expansion_pct_at_entry": float(getattr(sig, "vwap_stack_expansion_pct", 0.0)) if getattr(sig, "vwap_stack_expansion_pct", None) is not None else None,
            "vwap_stack_slope_pph_at_entry": float(getattr(sig, "vwap_stack_slope_pph", 0.0)) if getattr(sig, "vwap_stack_slope_pph", None) is not None else None,
        }

        try:
            pid = await self.db.insert_position(payload)
            LOG.info("Inserted position %s (id=%s)", sig.symbol, pid)

            # --- Protective orders (Bybit V5 conditional orders) ---
            sl_cid = create_stable_cid(pid, "SL")
            await self.exchange.create_order(
                sig.symbol, "market", "buy", actual_size, None,
                params={
                    "triggerPrice": float(stop_price),
                    "clientOrderId": sl_cid, "category": "linear",
                    "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": 1  # rising price triggers buy for short
                }
            )

            tp1_cid, tp_final_cid = None, None
            if self.cfg.get("PARTIAL_TP_ENABLED", False):
                tp1_cid = create_stable_cid(pid, "TP1")
                tp_price = actual_entry_price - float(self.cfg.get("PARTIAL_TP_ATR_MULT", 1.0)) * float(sig.atr)
                qty_tp1 = actual_size * float(self.cfg.get("PARTIAL_TP_PCT", 0.5))
                qty_tp1 = max(0.0, min(qty_tp1, actual_size))  # guard remainder
                await self.exchange.create_order(
                    sig.symbol, "market", "buy", qty_tp1, None,
                    params={
                        "triggerPrice": float(tp_price),
                        "clientOrderId": tp1_cid, "category": "linear",
                        "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": 2  # falling price triggers buy for short TP
                    }
                )
                if self.cfg.get("FINAL_TP_ENABLED", True):
                    tp_final_cid = create_stable_cid(pid, "TP_FINAL")
                    tp_final_price = actual_entry_price - float(self.cfg.get("FINAL_TP_ATR_MULT", 1.0)) * float(sig.atr)
                    remainder = max(0.0, actual_size - qty_tp1)
                    await self.exchange.create_order(
                        sig.symbol, "market", "buy", remainder, None,
                        params={
                            "triggerPrice": float(tp_final_price),
                            "clientOrderId": tp_final_cid, "category": "linear",
                            "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": 2
                        }
                    )
            elif self.cfg.get("FINAL_TP_ENABLED", True):
                tp_final_cid = create_stable_cid(pid, "TP_FINAL")
                tp_price = actual_entry_price - float(self.cfg.get("FINAL_TP_ATR_MULT", 1.0)) * float(sig.atr)
                await self.exchange.create_order(
                    sig.symbol, "market", "buy", actual_size, None,
                    params={
                        "triggerPrice": float(tp_price),
                        "clientOrderId": tp_final_cid, "category": "linear",
                        "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": 2
                    }
                )

            # --- Update DB with protective details ---
            await self.db.update_position(
                pid,
                status="OPEN",
                stop_price=float(stop_price),
                sl_cid=sl_cid,
                tp1_cid=tp1_cid,
                tp_final_cid=tp_final_cid,
            )

            # --- Mirror to in-memory map ---
            row = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)
            self.open_positions[pid] = dict(row)

            # --- Telegram notify ---
            vwap_frac  = getattr(sig, "vwap_stack_frac", None)
            vwap_exp   = getattr(sig, "vwap_stack_expansion_pct", None)  # already a percent value like 0.82%
            vwap_slope = getattr(sig, "vwap_stack_slope_pph", None)

            msg = (
                f"🔔 Opened SHORT {sig.symbol}\n"
                f"Entry: {actual_entry_price:.8f}\n"
                f"ATR(1h): {sig.atr:.8f}  ATR%: {sig.atr_pct:.2f}%\n"
                f"Risk: ${sig.risk_usd:.2f}  (ETH×{eth_mult:.2f} · VWAP×{vw_mult:.2f} · WP×{wp_mult:.2f})\n"
                f"VWAP stack: frac={vwap_frac if vwap_frac is not None else 'N/A'}  "
                f"exp={f'{vwap_exp:.2f}%' if vwap_exp is not None else 'N/A'}  "
                f"slope_pph={f'{vwap_slope:.4f}' if vwap_slope is not None else 'N/A'}\n"
                f"WinProb: {wp:.2%}"
            )
            await self.tg.send(msg)

        except Exception as e:
            # EMERGENCY CLOSE to prevent naked position
            msg = f"🚨 CRITICAL: Failed to persist/protect {sig.symbol}: {e}. Emergency closing now."
            LOG.critical(msg)
            await self.tg.send(msg)
            try:
                await self.exchange.create_market_order(
                    sig.symbol, "buy", actual_size, params={"reduceOnly": True, "category": "linear"}
                )
                await self.tg.send(f"✅ Emergency close filled for {sig.symbol}.")
            except Exception as close_e:
                await self.tg.send(f"🚨 FAILED EMERGENCY CLOSE for {sig.symbol}: {close_e}")

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
        symbol = pos["symbol"]

        # --- FIX: BOT-SIDE SAFETY NET ---
        # First, check the ultimate source of truth: the position size.
        try:
            positions = await self.exchange.fetch_positions(symbols=[symbol])
            position_size = 0.0
            if positions and positions[0]:
                position_size = float(positions[0].get('info', {}).get('size', 0))
            
            if position_size == 0:
                LOG.info("Position size for %s is 0. Trade is closed. Inferring exit reason and finalizing...", symbol)
                
                inferred_reason = "UNKNOWN"
                open_orders = await self._all_open_orders(symbol)
                open_cids = {o.get("clientOrderId") for o in open_orders}
                
                # --- FIX: More robust inference logic ---
                # First, check if any of the possible Take Profit orders are missing.
                if (pos.get("tp_final_cid") and pos["tp_final_cid"] not in open_cids) or \
                (pos.get("tp2_cid") and pos["tp2_cid"] not in open_cids) or \
                (pos.get("tp1_cid") and pos["tp1_cid"] not in open_cids):
                    inferred_reason = "TP"
                # If not a TP, check if any of the possible Stop Loss orders are missing.
                elif (pos.get("sl_trail_cid") and pos["sl_trail_cid"] not in open_cids) or \
                    (pos.get("sl_cid") and pos["sl_cid"] not in open_cids):
                    inferred_reason = "SL"
                
                LOG.info("Inferred exit reason for %s is: %s", symbol, inferred_reason)
                
                await self._finalize_position(pid, pos, inferred_exit_reason=inferred_reason)
                return
        except Exception as e:
            LOG.error("Could not fetch position size for %s during update: %s", symbol, e)
            return
        # --- END OF SAFETY NET ---

        orders = await self._all_open_orders(symbol)
        open_cids = {o.get("clientOrderId") for o in orders}

        if self.cfg.get("TIME_EXIT_ENABLED", cfg.TIME_EXIT_ENABLED):
            ddl = pos.get("exit_deadline")
            if ddl and datetime.now(timezone.utc) >= ddl:
                LOG.info("Time-exit firing on %s (pid %d)", symbol, pid)
                await self._force_close_position(pid, pos, tag="TIME_EXIT")
                return

        if self.cfg.get("PARTIAL_TP_ENABLED", False) and not pos["trailing_active"] and pos.get("tp1_cid") not in open_cids:
            fill_price = None
            try:
                o = await self._fetch_by_cid(pos["tp1_cid"], symbol)
                if o and o.get('status') == 'closed':
                    fill_price = o.get('average') or o.get('price')
            except Exception as e:
                LOG.warning("Failed to fetch TP1 fill price for %s: %s", pos["tp1_cid"], e)

            await self.db.add_fill(
                pid, "TP1", fill_price, float(pos["size"]) * self.cfg["PARTIAL_TP_PCT"], datetime.now(timezone.utc)
            )
            await self.db.update_position(pid, trailing_active=True)
            pos["trailing_active"] = True
            await self._activate_trailing(pid, pos)
            await self.tg.send(f"📈 TP1 hit on {symbol}, trailing activated")

        if pos["trailing_active"]:
            await self._trail_stop(pid, pos)

        active_stop_cid = pos.get("sl_trail_cid") if pos["trailing_active"] else pos.get("sl_cid")
        is_closed = active_stop_cid not in open_cids
        
        if not is_closed and pos["trailing_active"] and self.cfg.get("FINAL_TP_ENABLED", False):
            if pos.get("tp2_cid") not in open_cids:
                is_closed = True

        if is_closed:
            await self._finalize_position(pid, pos)

    async def _activate_trailing(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        try:
            if pos.get("sl_cid"):
                await self._cancel_by_cid(pos["sl_cid"], symbol)
        except ccxt.OrderNotFound:
            pass
        except Exception as e:
            LOG.warning(f"Could not cancel original SL for {pid} ({pos.get('sl_cid')}): {e}")

        # This places the first trailing stop order
        await self._trail_stop(pid, pos, first=True)

        # FIX: Correctly create, use, and persist the TP2 client order ID
        if self.cfg.get("FINAL_TP_ENABLED", False):
            try:
                final_tp_price = float(pos["entry_price"]) - self.cfg["FINAL_TP_ATR_MULT"] * float(pos["atr"])
                qty_left = float(pos["size"]) * (1 - self.cfg["PARTIAL_TP_PCT"])
                
                # 1. Create a stable, deterministic CID
                tp2_cid = create_stable_cid(pid, "TP2")
                
                await self.exchange.create_order(
                    symbol, "market", "buy", qty_left, None,
                    params={
                        "triggerPrice": final_tp_price, 
                        "clientOrderId": tp2_cid, 
                        'reduceOnly': True, 
                        'closeOnTrigger': True, 
                        'triggerDirection': 2, 
                        'category': 'linear'
                    }
                )
                
                # 2. Persist the new CID to the database
                await self.db.update_position(pid, tp2_cid=tp2_cid)
                
                # 3. Update the in-memory state to match
                pos["tp2_cid"] = tp2_cid
                LOG.info("Final Take Profit (TP2) placed for %s with CID %s", symbol, tp2_cid)

            except Exception as e:
                LOG.error(f"Failed to place final TP2 order for {pid}: {e}")

    async def _trail_stop(self, pid: int, pos: Dict[str, Any], first: bool = False):
        """
        Trailing BUY stop for a SHORT position:
        new_stop = price + k*ATR  (always ABOVE current price)
        Move it only in the favorable direction (downwards for shorts).
        """
        symbol = pos["symbol"]
        price = float((await self.exchange.fetch_ticker(symbol))["last"])
        atr = float(pos["atr"])
        prev_stop = float(pos.get("stop_price", 0) or 0.0)

        # Correct formula for shorts: stop ABOVE price
        new_stop = price + float(self.cfg.get("TRAIL_DISTANCE_ATR_MULT", 1.0)) * atr

        # Favorable move for shorts: lower stop (new_stop < prev_stop)
        favorable = (prev_stop == 0.0) or (new_stop < prev_stop)
        min_move = price * float(self.cfg.get("TRAIL_MIN_MOVE_PCT", 0.001))
        significant = abs(prev_stop - new_stop) > min_move

        if first or (favorable and significant):
            qty_left = float(pos["size"]) * (1 - float(self.cfg.get("PARTIAL_TP_PCT", 0.7)))
            sl_trail_cid = create_stable_cid(pid, "SL_TRAIL")

            # Cancel previous trailing SL if any
            try:
                if (not first) and pos.get("sl_trail_cid"):
                    await self._cancel_by_cid(pos["sl_trail_cid"], symbol)
            except ccxt.OrderNotFound:
                pass
            except Exception as e:
                LOG.warning("Trail cancel failed for %s: %s", symbol, e)
                return

            # Place new conditional BUY stop; trigger when price rises to it → triggerDirection=1
            await self.exchange.create_order(
                symbol, 'market', 'buy', qty_left, None,
                params={
                    "triggerPrice": new_stop,
                    "clientOrderId": sl_trail_cid, "category": "linear",
                    "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": 1
                }
            )
            await self.db.update_position(pid, stop_price=new_stop, sl_trail_cid=sl_trail_cid)
            pos["stop_price"] = new_stop
            pos["sl_trail_cid"] = sl_trail_cid
            LOG.info("Trail updated %s to %.6f", symbol, new_stop)

            
    async def _finalize_position(self, pid: int, pos: Dict[str, Any], inferred_exit_reason: str = None):
        symbol = pos["symbol"]
        opened_at = pos["opened_at"]
        entry_price = float(pos["entry_price"])
        size = float(pos["size"])
        side = (pos.get("side") or "short").lower()  # "short" or "long"
        side_mult = -1.0 if side == "short" else 1.0

        closed_at = datetime.now(timezone.utc)
        exit_price, exit_qty, closing_order_type = None, 0.0, inferred_exit_reason or "UNKNOWN"

        # --- 1) Identify closing order CID by reason (optional, best-effort) ---
        closing_order_cid = None
        if inferred_exit_reason == "TP":
            closing_order_cid = pos.get("tp_final_cid") or pos.get("tp2_cid") or pos.get("tp1_cid")
        elif inferred_exit_reason == "SL":
            closing_order_cid = pos.get("sl_trail_cid") or pos.get("sl_cid")

        # --- 2) Try to fetch a concrete exit order price (last chunk) ---
        if closing_order_cid:
            try:
                order = await self._fetch_by_cid(closing_order_cid, symbol)
                if order and (order.get("average") or order.get("price")):
                    exit_price = float(order.get("average") or order.get("price"))
                    exit_qty = float(order.get("filled", 0) or 0)
                    closing_order_type = inferred_exit_reason
            except Exception as e:
                LOG.warning("Could not fetch closing order by CID %s for %s: %s", closing_order_cid, symbol, e)

        # --- 3) Robust fallback: use recent trade(s) as ground truth for the last chunk ---
        if not exit_price:
            try:
                await asyncio.sleep(1.5)  # let exchange settle fills
                my_trades = await self.exchange.fetch_my_trades(symbol, limit=10)
                # If SHORT, the final close is a BUY; if LONG, it's a SELL
                close_side = "buy" if side == "short" else "sell"
                closing_trade = next((t for t in reversed(my_trades) if (t.get("side") or "").lower() == close_side), None)
                if closing_trade:
                    exit_price = float(closing_trade["price"])
                    exit_qty = float(closing_trade.get("amount") or 0)
                    if closing_order_type == "UNKNOWN":
                        closing_order_type = "FALLBACK_FILL"
                else:
                    LOG.error("CRITICAL: No closing trade found for %s. Using entry price as fallback.", symbol)
                    exit_price, exit_qty = entry_price, size
            except Exception as e:
                LOG.error("CRITICAL: fetch_my_trades fallback failed for %s: %s. Using entry price.", symbol, e)
                exit_price, exit_qty = entry_price, size

        # Record the last known fill (does not double-count; we aggregate below)
        try:
            await self.db.add_fill(pid, closing_order_type, float(exit_price), float(exit_qty), closed_at)
        except Exception as e:
            LOG.warning("Could not persist closing fill for %s: %s", symbol, e)

        # --- 4) Aggregate PnL correctly from ALL fills (handles TP1/TP2/trailing) ---
        # Assumption: entries were recorded as one SELL fill (short) or one BUY fill (long) at entry.
        # Exits are recorded as BUY fills (short) or SELL fills (long).
        rows = await self.db.pool.fetch("SELECT kind, price, qty FROM fills WHERE position_id=$1 ORDER BY ts ASC", pid)

        # Compute weighted prices by side:
        # For SHORT:  entry is SELL (we opened size at entry_price), exits are BUYs at various prices.
        # PnL_short = Σ (entry_price * qty_entry) - Σ (exit_price * qty_exit)
        # For LONG:   PnL_long  = Σ (exit_price * qty_exit) - Σ (entry_price * qty_entry)
        entry_notional = 0.0
        exit_notional = 0.0
        entry_qty = 0.0
        exit_qty_sum = 0.0

        for r in rows:
            px = float(r["price"] or 0.0)
            q  = float(r["qty"] or 0.0)
            k  = (r["kind"] or "").upper()

            # Heuristic: anything starting with "TP", "SL", "TIME_EXIT", "FALLBACK" treated as exit.
            is_exit_kind = (k.startswith("TP") or k.startswith("SL") or "TIME_EXIT" in k or "FALLBACK" in k)
            if is_exit_kind:
                exit_notional += px * q
                exit_qty_sum  += q
            else:
                # treat as entry (you likely recorded a single fill when position opened)
                entry_notional += px * q
                entry_qty      += q

        # If entry fills weren’t recorded, fall back to position snapshot
        if entry_qty <= 0.0:
            entry_notional = entry_price * size
            entry_qty = size

        # Sanity: ensure we closed roughly the whole size
        if abs(exit_qty_sum - entry_qty) > 1e-8:
            LOG.warning("Position %s fills mismatch: exit_qty %.6f != entry_qty %.6f", symbol, exit_qty_sum, entry_qty)

        if side == "short":
            total_pnl = entry_notional - exit_notional
            avg_exit_price = (exit_notional / exit_qty_sum) if exit_qty_sum > 0 else entry_price
            pnl_pct = (entry_price / avg_exit_price - 1.0) * 100.0 if avg_exit_price > 0 else 0.0
        else:
            total_pnl = exit_notional - entry_notional
            avg_exit_price = (exit_notional / exit_qty_sum) if exit_qty_sum > 0 else entry_price
            pnl_pct = (avg_exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

        holding_minutes = (closed_at - opened_at).total_seconds() / 60 if opened_at else 0.0

        # --- 5) Post-trade analytics (your original logic, kept) ---
        mae_usd = mfe_usd = mae_over_atr = mfe_over_atr = 0.0
        realized_vol_during_trade = btc_beta_during_trade = 0.0
        try:
            since_ts = int(opened_at.timestamp() * 1000)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, "1m", since=since_ts)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df[df["ts"] <= closed_at]
                if not df.empty:
                    if side == "short":
                        mae_usd = max(0.0, (df["h"].max() - entry_price) * entry_qty)
                        mfe_usd = max(0.0, (entry_price - df["l"].min()) * entry_qty)
                    else:
                        mae_usd = max(0.0, (entry_price - df["l"].min()) * entry_qty)
                        mfe_usd = max(0.0, (df["h"].max() - entry_price) * entry_qty)

                    asset_returns = df["c"].pct_change().dropna()
                    if not asset_returns.empty:
                        realized_vol_during_trade = asset_returns.std() * (365 * 24 * 60) ** 0.5

                    benchmark_symbol = self.cfg.get("REGIME_BENCHMARK_SYMBOL", "BTCUSDT")
                    if symbol != benchmark_symbol:
                        btc_ohlcv = await self.exchange.fetch_ohlcv(benchmark_symbol, "1m", since=since_ts)
                        if btc_ohlcv:
                            bdf = pd.DataFrame(btc_ohlcv, columns=["ts","o","h","l","c","v"])
                            bdf["ts"] = pd.to_datetime(bdf["ts"], unit="ms", utc=True)
                            bdf = bdf[bdf["ts"] <= closed_at]
                            if not bdf.empty:
                                comb = pd.concat([asset_returns.rename("asset"),
                                                bdf["c"].pct_change().rename("btc")], axis=1).dropna()
                                if len(comb) > 5 and comb["btc"].var() > 0:
                                    btc_beta_during_trade = comb["asset"].cov(comb["btc"]) / comb["btc"].var()

            atr_at_entry = float(pos.get("atr") or 0.0)
            if atr_at_entry > 0:
                mae_over_atr = (mae_usd / entry_qty) / atr_at_entry
                mfe_over_atr = (mfe_usd / entry_qty) / atr_at_entry
        except Exception as e:
            LOG.warning("Post-trade metrics failed for %s: %s", symbol, e)

        # --- 6) Cleanup + persist ---
        try:
            await self.exchange.cancel_all_orders(symbol, params={"category": "linear"})
        except Exception as e:
            LOG.warning("Final cleanup for position %d (%s) failed: %s", pid, symbol, e)

        await self.db.update_position(
            pid, status="CLOSED", closed_at=closed_at, pnl=total_pnl,
            exit_reason=closing_order_type, holding_minutes=holding_minutes,
            pnl_pct=pnl_pct, mae_usd=mae_usd, mfe_usd=mfe_usd,
            mae_over_atr=mae_over_atr, mfe_over_atr=mfe_over_atr,
            realized_vol_during_trade=realized_vol_during_trade,
            btc_beta_during_trade=btc_beta_during_trade
        )

        await self.risk.on_trade_close(total_pnl, self.tg)
        self.open_positions.pop(pid, None)
        self.last_exit[symbol] = closed_at
        await self.tg.send(f"✅ {symbol} position closed. Total PnL ≈ {total_pnl:.2f} USDT")


    async def _force_open_position(self, symbol: str):
        """
        Manually triggers a trade for a given symbol for testing purposes.
        Runs a full scan to gather real data but still respects veto filters.
        """
        await self.tg.send(f"Received force open command for {symbol}. Attempting to generate signal data...")
        LOG.info("Manual trade requested via Telegram for symbol: %s", symbol)

        if any(p['symbol'] == symbol for p in self.open_positions.values()):
            msg = f"⚠️ Cannot force open {symbol}: A position is already open for this symbol."
            LOG.warning(msg); await self.tg.send(msg); return

        try:
            current_market_regime = await self.regime_detector.get_current_regime()
            # Reuse the same ETH barometer logic as main loop (no recompute inside scan)
            eth_macd_data = None
            try:
                eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=100)
                if eth_ohlcv:
                    df_eth = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    macd_df = ta.macd(df_eth['close'])
                    latest_macd = macd_df.iloc[-1]
                    eth_macd_data = {"macd": latest_macd['macd'], "signal": latest_macd['signal'], "hist": latest_macd['hist']}
            except Exception as e:
                LOG.warning(f"Could not calculate ETH MACD barometer: {e}")

            # ✅ PASS THE THIRD ARG
            signal = await self._scan_symbol_for_signal(symbol, current_market_regime, eth_macd_data)
            if signal is None:
                msg = f"❌ Failed to force open {symbol}: Could not generate signal data."
                LOG.error(msg); await self.tg.send(msg); return

            equity = await self.db.latest_equity() or 0.0
            open_positions_count = len(self.open_positions)
            ok, vetoes = filters.evaluate(
                signal, listing_age_days=signal.listing_age_days,
                open_positions=open_positions_count, equity=equity
            )
            if not ok:
                msg = f"⚠️ VETOED force open for {symbol}: {' | '.join(vetoes)}"
                LOG.warning(msg); await self.tg.send(msg); return

            LOG.info("Bypassing signal conditions and proceeding to open position for %s.", symbol)
            await self.tg.send(f"✅ Signal data generated for {symbol}. Proceeding with forced entry.")
            await self._open_position(signal)

        except Exception as e:
            msg = f"❌ An unexpected error occurred during force open for {symbol}: {e}"
            LOG.error(msg, exc_info=True); await self.tg.send(msg)


    async def _force_close_position(self, pid: int, pos: Dict[str, Any], tag: str):
            symbol = pos["symbol"]
            size = float(pos["size"])
            entry_price = float(pos["entry_price"])
            
            try:
                # 1. Cancel any other open orders (like SL/TP) for this position
                await self.exchange.cancel_all_orders(symbol)
                
                # 2. Send the market order to close the position
                await self.exchange.create_market_order(symbol, "buy", size, params={'reduceOnly': True})
                LOG.info(f"Force-closing {symbol} (pid {pid}) with a market order due to: {tag}")

            except Exception as e:
                LOG.warning("Force-close order issue on %s: %s", symbol, e)

            # --- THIS IS THE CRITICAL FIX ---
            # 3. Wait a moment for the trade to register on the exchange
            await asyncio.sleep(2) # 2-second buffer

            # 4. Fetch the actual fill price from our trade history
            exit_price = None
            try:
                # Fetch our most recent trades for this symbol
                my_trades = await self.exchange.fetch_my_trades(symbol, limit=5)
                # The closing trade is the most recent 'buy' order
                closing_trade = next((t for t in reversed(my_trades) if t.get('side') == 'buy'), None)
                
                if closing_trade:
                    exit_price = float(closing_trade['price'])
                    LOG.info(f"Confirmed force-close fill price for {symbol} at {exit_price}")
                else:
                    LOG.warning(f"Could not confirm fill price for force-close of {symbol} via fetch_my_trades. Falling back to ticker.")
            except Exception as e:
                LOG.error(f"Error fetching fill price for force-close of {symbol}: {e}")

            # 5. Last resort fallback (should rarely be used now)
            if not exit_price:
                exit_price = float((await self.exchange.fetch_ticker(symbol))["last"])

            # 6. Calculate PnL with the accurate price
            pnl = (entry_price - exit_price) * size
            closed_at = datetime.now(timezone.utc)
            holding_minutes = (closed_at - pos["opened_at"]).total_seconds() / 60 if pos.get("opened_at") else 0.0
            
            # 7. Update the database with all the correct information
            await self.db.update_position(
                pid, 
                status="CLOSED", 
                closed_at=closed_at, 
                pnl=pnl,
                exit_reason=tag, # Use the tag (e.g., "TIME_EXIT") as the reason
                holding_minutes=holding_minutes
            )
            # --- END OF FIX ---
            
            await self.db.add_fill(pid, tag, exit_price, size, closed_at)
            await self.risk.on_trade_close(pnl, self.tg)
            self.last_exit[symbol] = closed_at
            self.open_positions.pop(pid, None)
            
            await self.tg.send(f"⏰ {symbol} closed by {tag}. PnL ≈ {pnl:.2f} USDT")

    async def _main_signal_loop(self):
        LOG.info("Starting main signal scan loop.")
        while True:
            try:
                if self.paused or not self.risk.can_trade():
                    await asyncio.sleep(5)
                    continue

                current_market_regime = await self.regime_detector.get_current_regime()
                LOG.info("Starting new scan cycle for %d symbols with market regime: %s", len(self.symbols), current_market_regime)

                # --- NEW: Calculate the ETH MACD Barometer ---
                eth_macd_data = None
                try:
                    # Fetch 4-hour data for ETHUSDT
                    eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=100)
                    if eth_ohlcv:
                        df_eth = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        # Calculate MACD using your indicators module
                        macd_df = ta.macd(df_eth['close'])
                        # Get the very last (most recent) row of MACD data
                        latest_macd = macd_df.iloc[-1]
                        eth_macd_data = {
                            "macd": latest_macd['macd'],
                            "signal": latest_macd['signal'],
                            "hist": latest_macd['hist']
                        }
                        LOG.info(f"ETH Barometer MACD(4h): {latest_macd['macd']:.2f}, Hist: {latest_macd['hist']:.2f}")
                except Exception as e:
                    LOG.warning(f"Could not calculate ETH MACD barometer: {e}")
                # --- END OF NEW SECTION ---

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
                        # FIX 1.4: This is the single, correct pre-flight check location
                        try:
                            positions = await self.exchange.fetch_positions(symbols=[sym])
                            if positions and positions[0] and float(positions[0].get('info', {}).get('size', 0)) > 0:
                                LOG.info("Skipping scan for %s, pre-flight check found position already exists on exchange.", sym)
                                if sym not in open_symbols:
                                    LOG.warning("ORPHAN POSITION DETECTED for %s during pre-flight check! Reconciliation needed.", sym)
                                continue
                        except Exception as e:
                            LOG.error("Could not perform pre-flight position check for %s: %s", sym, e)
                            continue

                        signal = await self._scan_symbol_for_signal(sym, current_market_regime, eth_macd_data)
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

        elif root == "/report" and len(parts) == 2:
            period = parts[1].lower() # e.g., 'daily', 'weekly', '6h'
            if period in ['6h', 'daily', 'weekly', 'monthly']:
                await self.tg.send(f"Generating on-demand '{period}' report...")
                summary_text = await self._generate_summary_report(period)
                await self.tg.send(summary_text)
            else:
                await self.tg.send(f"Unknown period '{period}'. Use: 6h, daily, weekly, monthly.")        

        elif root == "/resume":
            if self.risk.can_trade():
                self.paused = False
                await self.tg.send("▶️ Resumed")
            else:
                await self.tg.send("⚠️ Kill switch active")
        elif root == "/set" and len(parts) == 3:
            raw_key, val = parts[1], parts[2]
            key = self._normalize_cfg_key(raw_key)
            try:
                cast = json.loads(val)   # allows numbers, true/false, etc.
            except json.JSONDecodeError:
                cast = val
            self.cfg[key] = cast
            await self.tg.send(f"✅ {raw_key} → {key} set to {cast}")
            return

        elif root == "/open" and len(parts) == 2:
            symbol = parts[1].upper() # e.g., BTCUSDT
            # We run this as a background task so it doesn't block the Telegram loop
            # while it fetches data and opens the trade.
            asyncio.create_task(self._force_open_position(symbol))
            return # Acknowledge the command immediately

        elif root == "/status":
            await self.tg.send(json.dumps({
                "paused": self.paused,
                "open": len(self.open_positions),
                "loss_streak": self.risk.loss_streak
            }, indent=2))

        elif root == "/analyze":
            await self.tg.send("🤖 Roger that. Starting the full analysis process. This requires the `bybit.csv` file to be present. The process may take a few minutes. I will send the report file here when complete.")
            
            # Run the master script as a non-blocking background process
            try:
                subprocess.Popen(["/opt/livefader/src/run_weekly_report.sh"])
            except Exception as e:
                await self.tg.send(f"❌ Failed to start the analysis script: {e}")
            return

    async def _resume(self):
            """
            On startup, intelligently load state from DB and reconcile with the exchange.
            This version identifies valid positions FIRST, then cleans up only true orphans,
            and includes all necessary reconciliation checks.
            """
            LOG.info("--> Resuming state with intelligent reconciliation...")

            # --- STEP 1: Fetch current state from both Exchange and Database ---
            LOG.info("Step 1: Fetching open positions from EXCHANGE and DATABASE...")
            try:
                exchange_positions = await self.exchange.fetch_positions()
                open_exchange_positions = {
                    p['info']['symbol']: p for p in exchange_positions if float(p['info'].get('size', 0)) > 0
                }
                LOG.info("...Success! Found %d positions on the exchange.", len(open_exchange_positions))
            except Exception as e:
                LOG.error("CRITICAL: Could not fetch exchange positions on startup: %s. Exiting.", e)
                sys.exit(1)

            db_positions_rows = await self.db.fetch_open_positions()
            db_positions = {r["symbol"]: dict(r) for r in db_positions_rows}
            LOG.info("...Success! Found %d 'OPEN' positions in the database.", len(db_positions))

            # --- STEP 2: Reconcile positions and identify all VALID client order IDs ---
            LOG.info("Step 2: Reconciling positions and identifying valid protective orders...")
            valid_cids = set()
            now_utc = datetime.now(timezone.utc)

            # Check for orphan positions on the exchange
            for symbol, pos_data in open_exchange_positions.items():
                if symbol not in db_positions:
                    msg = f"🚨 ORPHAN DETECTED: Position for {symbol} exists on exchange but not in DB. Closing immediately."
                    LOG.warning(msg)
                    await self.tg.send(msg)
                    try:
                        side = 'buy' if pos_data['side'] == 'short' else 'sell'
                        size = float(pos_data['info']['size'])
                        await self.exchange.create_market_order(symbol, side, size, params={'reduceOnly': True})
                    except Exception as e:
                        LOG.error("Failed to force-close orphan position %s: %s", symbol, e)

            # Check for DB/exchange mismatches and build the "safe list" of CIDs
            for symbol, pos_row in list(db_positions.items()):
                pid = pos_row["id"]
                
                if symbol not in open_exchange_positions:
                    LOG.warning(f"DB/EXCHANGE MISMATCH: Position {pid} for {symbol} is 'OPEN' in DB but not on exchange. Marking as closed.")
                    await self.db.update_position(pid, status="CLOSED", closed_at=now_utc, pnl=0, exit_reason="RECONCILE_CLOSE")
                    del db_positions[symbol]
                    continue

                # --- NEW: Check for stale positions on restart ---
                opened_at = pos_row.get("opened_at")
                max_holding_duration = None
                if self.cfg.get("TIME_EXIT_HOURS_ENABLED", False):
                    max_holding_duration = timedelta(hours=self.cfg.get("TIME_EXIT_HOURS", 4))
                elif self.cfg.get("TIME_EXIT_ENABLED", False):
                    max_holding_duration = timedelta(days=self.cfg.get("TIME_EXIT_DAYS", 10))

                if opened_at and max_holding_duration and (now_utc - opened_at) > max_holding_duration:
                    msg = f"⏰ STALE POSITION DETECTED on restart: {symbol} (pid {pid}) is older than the time limit. Forcing close."
                    LOG.warning(msg)
                    await self.tg.send(msg)
                    # Use create_task to close in the background without blocking startup
                    asyncio.create_task(self._force_close_position(pid, pos_row, tag="STALE_ON_RESTART"))
                    # Remove from the list of positions to be loaded into memory
                    del db_positions[symbol]
                    continue # Skip to the next position

                # --- END OF NEW LOGIC ---

                ex_pos = open_exchange_positions[symbol]
                db_size = float(pos_row['size'])
                ex_size = float(ex_pos['info']['size'])
                if abs(db_size - ex_size) > 1e-9:
                    msg = (f"🚨 SIZE MISMATCH on resume for {symbol} (pid {pid}): DB size is {db_size}, Exchange size is {ex_size}. Flagging for manual review.")
                    LOG.critical(msg)
                    await self.tg.send(msg)
                    await self.db.update_position(pid, status="SIZE_MISMATCH")
                    del db_positions[symbol]
                    continue

                LOG.debug("Identified valid CIDs for position %s to preserve.", symbol)
                if pos_row.get("sl_cid"): valid_cids.add(pos_row["sl_cid"])
                if pos_row.get("tp1_cid"): valid_cids.add(pos_row["tp1_cid"])
                if pos_row.get("tp_final_cid"): valid_cids.add(pos_row["tp_final_cid"])
                if pos_row.get("sl_trail_cid"): valid_cids.add(pos_row["sl_trail_cid"])

            LOG.info("...Identified %d CIDs belonging to valid positions.", len(valid_cids))

            # --- STEP 3: Perform the INTELLIGENT Clean Slate Protocol ---
            LOG.info("Step 3: Fetching all open orders to clean up ORPHANS ONLY...")
            try:
                all_open_orders = await self._all_open_orders_for_all_symbols()
                
                cancel_tasks = []
                for order in all_open_orders:
                    cid = order.get("clientOrderId")
                    if cid and cid.startswith("bot_") and cid not in valid_cids:
                        LOG.warning("Found orphaned order %s for symbol %s. Scheduling for cancellation.", cid, order['symbol'])
                        cancel_tasks.append(self.exchange.cancel_order(order['id'], order['symbol'], params={'category': 'linear'}))
                
                if cancel_tasks:
                    LOG.info("Cancelling %d orphaned orders...", len(cancel_tasks))
                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                    LOG.info("...Orphaned order cleanup complete.")
                else:
                    LOG.info("...No orphaned orders found to cancel.")
                    
            except Exception as e:
                LOG.error("CRITICAL: Failed to perform clean slate protocol on startup: %s", e)
                traceback.print_exc()

            # --- STEP 4: Load final state into memory ---
            LOG.info("Step 4: Loading final reconciled positions into memory...")
            # The db_positions dict now only contains valid, non-stale positions
            for symbol, pos_row in db_positions.items():
                self.open_positions[pos_row["id"]] = pos_row
                LOG.info("...Successfully resumed and verified open position for %s (ID: %d)", symbol, pos_row["id"])

            LOG.info("Step 5: Fetching peak equity from DB...")
            peak = await self.db.pool.fetchval("SELECT MAX(equity) FROM equity_snapshots")
            self.peak_equity = float(peak) if peak is not None else 0.0
            LOG.info("...Initial peak equity loaded: $%.2f", self.peak_equity)

            LOG.info("Step 6: Loading recent exit timestamps for cooldowns...")
            cd_h = int(self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS))
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
        await self.db.migrate_schema()

        if self.settings.bybit_testnet:
            LOG.warning("="*60)
            LOG.warning("RUNNING ON TESTNET")
            LOG.warning("Testnet data is unreliable for most altcoins.")
            LOG.warning("="*60)

        LOG.info("Loading exchange market data...")
        try:
            await self.exchange._exchange.load_markets()
            LOG.info("Market data loaded.")
        except Exception as e:
            LOG.error("Could not load markets: %s. Exiting.", e)
            return

        LOG.info("Loading symbol listing dates...")
        self._listing_dates_cache = await self._load_listing_dates()

        await self._resume()
        await self.tg.send("🤖 LIVEFADER v7.0")

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._main_signal_loop())
                tg.create_task(self._manage_positions_loop())
                tg.create_task(self._telegram_loop())
                tg.create_task(self._equity_loop())
                tg.create_task(self._reporting_loop())
        except* (asyncio.CancelledError, KeyboardInterrupt):
            LOG.info("Shutdown signal received.")
        finally:
            await self.exchange.close()
            if self.db.pool:
                await self.db.pool.close()
            await self.tg.close()
            LOG.info("Bot shut down cleanly.")

    async def _generate_summary_report(self, period: str) -> str:
        """
        Queries the database for a given period, calculates KPIs, and returns a formatted summary string.
        
        Args:
            period: A string like '6h', 'daily', 'weekly', or 'monthly'.
        """
        now = datetime.now(timezone.utc)
        period_map = {
            '6h': timedelta(hours=6),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30) # Approximation for monthly
        }
        
        if period not in period_map:
            return f"Error: Unknown report period '{period}'."

        start_time = now - period_map[period]
        LOG.info("Generating %s summary report for trades closed since %s", period, start_time.isoformat())

        try:
            query = """
                SELECT
                    pnl
                FROM positions
                WHERE status = 'CLOSED' AND closed_at >= $1
            """
            records = await self.db.pool.fetch(query, start_time)

            if not records:
                return f"📊 *{period.capitalize()} Report*\n\nNo trades were closed in the last {period}."

            total_trades = len(records)
            pnl_values = [float(r['pnl']) for r in records if r['pnl'] is not None]
            
            wins = [p for p in pnl_values if p > 0]
            losses = [p for p in pnl_values if p < 0]
            
            win_count = len(wins)
            loss_count = len(losses)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = sum(pnl_values)
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = sum(wins) / win_count if win_count > 0 else 0
            avg_loss = sum(losses) / loss_count if loss_count > 0 else 0
            
            expectancy = (avg_win * (win_rate / 100)) - (abs(avg_loss) * (1 - (win_rate / 100)))

            # Using MarkdownV2 for formatting
            report_lines = [
                f"📊 *{period.capitalize()} Performance Summary*",
                f"```{'-'*25}",
                f" Period: Last {period}",
                f" Total Closed Trades: {total_trades}",
                f" Total PnL: {total_pnl:+.2f} USDT",
                f"",
                f" Win Rate: {win_rate:.2f}% ({win_count} W / {loss_count} L)",
                f" Profit Factor: {profit_factor:.2f}",
                f" Expectancy/Trade: {expectancy:+.2f} USDT",
                f"",
                f" Avg Win:  {avg_win:+.2f} USDT",
                f" Avg Loss: {avg_loss:+.2f} USDT",
                f"```{'-'*25}",
            ]
            
            return "\n".join(report_lines)

        except Exception as e:
            LOG.error("Failed to generate summary report: %s", e)
            return f"Error: Could not generate {period} report. Check logs."

    async def _reporting_loop(self):
        """A background loop that sends scheduled reports to Telegram."""
        LOG.info("Reporting loop started.")
        last_report_sent = {} # Stores the last time a report was sent for a period

        while True:
            await asyncio.sleep(60 * 5) # Check every 5 minutes
            now = datetime.now(timezone.utc)
            
            periods_to_check = {
                '6h': now.hour % 6 == 0,
                'daily': now.hour == 0,
                'weekly': now.weekday() == 0 and now.hour == 0, # Monday morning
            }
            
            for period, should_send in periods_to_check.items():
                # To prevent spam on restart, we check if the last report for this period
                # was already sent within the current time window.
                last_sent_date = last_report_sent.get(period)
                
                # For daily/weekly, we just check the date. For 6h, we check the hour block.
                is_already_sent = False
                if period in ['daily', 'weekly'] and last_sent_date == now.date():
                    is_already_sent = True
                elif period == '6h' and last_sent_date == (now.date(), now.hour // 6):
                    is_already_sent = True

                if should_send and not is_already_sent:
                    LOG.info("Triggering scheduled '%s' report.", period)
                    summary_text = await self._generate_summary_report(period)
                    await self.tg.send(summary_text)
                    
                    # Mark as sent for this time window
                    if period in ['daily', 'weekly']:
                        last_report_sent[period] = now.date()
                    elif period == '6h':
                        last_report_sent[period] = (now.date(), now.hour // 6)

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