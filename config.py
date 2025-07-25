# config.py
"""
Strategy & risk defaults.
Edit in Git – for one‑off tweaks use config.yaml or Telegram `/set KEY value`.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV_DIR  = PROJECT_ROOT / "raw_csv"
PARQUET_DIR  = PROJECT_ROOT / "parquet"
SIGNALS_DIR  = PROJECT_ROOT / "signals"
RESULTS_DIR  = PROJECT_ROOT / "results"
SYMBOLS_FILE = PROJECT_ROOT / "symbols.txt"
MERGED_CSV_DIR = PROJECT_ROOT / "data_merged"

BYBIT_ACCOUNT_TYPE    = "UNIFIED"  # or "STANDARD"

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"

EMA_FAST_PERIOD = 20   # For 4h EMA cross
EMA_SLOW_PERIOD = 200  # For 4h EMA cross
VWAP_LEN = 15

PRICE_BOOM_PERIOD_H = 24
PRICE_BOOM_PCT      = 0.15
PRICE_SLOWDOWN_PERIOD_H = 4
PRICE_SLOWDOWN_PCT      = 0.01

# ── risk sizing ────────────────────────────────────────────────────────────
RISK_MODE          = "FIXED"     # or "FIXED"
FIXED_RISK_USDT    = 10.0
RISK_PCT           = 0.005         # 0.5 %

# ── stop / targets ─────────────────────────────────────────────────────────
SL_ATR_MULT             = 2.5
FINAL_TP_ENABLED        = True
FINAL_TP_ATR_MULT       = 1.0
PARTIAL_TP_ENABLED      = False
PARTIAL_TP_PCT          = 0.70       # 70 % size closed at TP1
PARTIAL_TP_ATR_MULT     = 1.0
TRAIL_ENABLED           = False
TRAIL_START_ATR_MULT    = 1.0
TRAIL_DISTANCE_ATR_MULT = 1.0
TRAIL_MIN_MOVE_PCT      = 0.001
TIME_EXIT_ENABLED       = True     # set False to disable
TIME_EXIT_DAYS          = 10         # close runner after N days

# --- Structural Trend Filter (Long-Term) ---
STRUCTURAL_TREND_DAYS    = 30
STRUCTURAL_TREND_RET_PCT = 0.01

# --- Sideways "Gap" Filter (Pre-Entry Consolidation) ---
GAP_VWAP_HOURS  = 2
GAP_MAX_DEV_PCT = 0.01
GAP_MIN_BARS    = 3

# --- Volume Filter (Applied during Scouting) ---
VOL_FILTER_ENABLED  = False
MIN_VOL_USD         = 10000    # Min 4-hour USD volume
MAX_VOL_USD         = 5000000000  # Max 4-hour USD volume

# --- Volatility Filter (Applied during Scouting) ---
VOLATILITY_FILTER_ENABLED = False
MIN_ATR_PCT               = 0.005 # Min ATR as a percentage of price

# ── portfolio / kill‑switches ─────────────────────────────────────────────
MAX_OPEN            = 30          # max concurrent positions
MAX_LOSS_STREAK     = 5
KILL_EQUITY_LEVEL   = 90.0        # USDT
MAX_LEVERAGE        = 10
MIN_NOTIONAL        = 0.01         # exchange min order size
MIN_STOP_DIST_USD   = 0.1
MIN_STOP_DIST_PCT   = 0.0008      # 0.08 %

# draw‑down pause
DD_PAUSE_ENABLED    = True     # set False to disable drawdown protection  # <--- ADD THIS
DD_MAX_PCT          = 50.0     # Pause trading if equity drops 10% from peak # <--- ADD THIS
DD_COOLDOWN_PCT     = 50.0     # % from equity peak
DD_COOLDOWN_DURATION_H = 12


# ── BTC / ALT EMA trend veto ──────────────────────────────────────────────
BTC_FAST_FILTER_ENABLED = False
BTC_FAST_TIMEFRAME      = "5m"
BTC_FAST_EMA_PERIOD     = 20
BTC_SLOW_FILTER_ENABLED = False
BTC_SLOW_TIMEFRAME      = "1h"
BTC_SLOW_EMA_PERIOD     = 200

ALT_SYMBOL              = "ETHUSDT"
ALT_FAST_FILTER_ENABLED = False
ALT_FAST_TIMEFRAME      = "5m"
ALT_FAST_EMA_PERIOD     = 20
ALT_SLOW_FILTER_ENABLED = False
ALT_SLOW_TIMEFRAME      = "1h"
ALT_SLOW_EMA_PERIOD     = 200

# ── Boom‑Volume rule ───────────────────────────────────────────────────────
BV_RULE_ENABLED   = False
BV_BUCKETS        = 5             # 1..5 buckets from tester
BV_MIN_VOL_SPIKE  = [1.2, 1.4, 1.6, 1.8, 2.0]
BV_MAX_VOL_SPIKE  = [6.0, 4.5, 3.5, 2.5, 2.0]

# ── Open‑interest confirmation ────────────────────────────────────────────
OI_FILTER_ENABLED        = False
OI_LOOKBACK_PERIOD_BARS  = 6       # history bars stored in cache +1
OI_MIN_CHANGE_PCT        = -5.0
OI_MAX_CHANGE_PCT        = 10.0

# ── runtime / misc ────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 60
ENTRY_ORDER_TYPE  = "MARKET"        # or "MARKET"

# ─── Re‑entry cool‑down per symbol ──────────────────────────────────────────
SYMBOL_COOLDOWN_HOURS = 4    # skip new signals for the same symbol during X h
