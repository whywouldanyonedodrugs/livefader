# config.py
"""
Strategy & risk defaults.
Edit in Git – for one‑off tweaks use config.yaml or Telegram `/set KEY value`.
"""

SYMBOLS_FILE = "symbols.txt"
START_DATE = "2022-01-01"
END_DATE = "2026-01-01"

EMA_FAST = 20   # For 4h EMA cross
EMA_SLOW = 200  # For 4h EMA cross

PRICE_BOOM_PERIOD_H = 24
PRICE_BOOM_PCT      = 0.15
PRICE_SLOWDOWN_PERIOD_H = 4
PRICE_SLOWDOWN_PCT      = 0.01

# ── risk sizing ────────────────────────────────────────────────────────────
RISK_MODE          = "FIXED"     # or "FIXED"
FIXED_RISK_USDT    = 10.0
RISK_PCT           = 0.005         # 0.5 %

# ── stop / targets ─────────────────────────────────────────────────────────
SL_ATR_MULT             = 3.0
FINAL_TP_ENABLED        = True
FINAL_TP_ATR_MULT       = 1.0
PARTIAL_TP_ENABLED      = False
PARTIAL_TP_PCT          = 0.70       # 70 % size closed at TP1
PARTIAL_TP_ATR_MULT     = 1.0
TRAIL_ENABLED           = False
TRAIL_START_ATR_MULT    = 1.0
TRAIL_DISTANCE_ATR_MULT = 1.0
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

# ── basic signal filters ──────────────────────────────────────────────────
RSI_RANGE       = (30, 80)
ATR_RANGE       = (0.0, 1.0)
REGIME_MA_PERIOD        = 100        # Long-term moving average period for trend
REGIME_ATR_PERIOD       = 20         # ATR period for Keltner Channel
REGIME_ATR_MULT         = 1.5        # ATR multiplier for Keltner Channel
REGIME_FILTER_ENABLED = False
ALLOWED_REGIMES = [
    "BULL_LOW_VOL",
    "BULL_HIGH_VOL",
    "BEAR_LOW_VOL",
    "BEAR_HIGH_VOL",
]

# ── portfolio / kill‑switches ─────────────────────────────────────────────
MAX_OPEN            = 30          # max concurrent positions
MAX_LOSS_STREAK     = 5
KILL_EQUITY_LEVEL   = 90.0        # USDT
MAX_LEVERAGE        = 10
MIN_NOTIONAL        = 0.01         # exchange min order size
MIN_STOP_DIST_USD   = 0.1
MIN_STOP_DIST_PCT   = 0.0008      # 0.08 %

# draw‑down pause
DD_COOLDOWN_PCT     = 50.0        # % from equity peak
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
