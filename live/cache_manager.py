# live/cache_manager.py
import sqlite3
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)
CACHE_DB_PATH = Path("atr_cache.db")

def init_cache():
    """Creates the SQLite database and table if they don't exist."""
    try:
        con = sqlite3.connect(CACHE_DB_PATH)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS atr_cache (
                symbol TEXT PRIMARY KEY,
                atr REAL,
                timestamp INTEGER
            )
        """)
        con.commit()
        con.close()
        LOG.info("ATR cache initialized successfully at %s", CACHE_DB_PATH)
    except Exception as e:
        LOG.error("Failed to initialize ATR cache: %s", e)

def save_atr_to_cache(symbol: str, atr: float, timestamp: float):
    """Saves or updates a symbol's ATR value in the cache."""
    try:
        con = sqlite3.connect(CACHE_DB_PATH)
        cur = con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO atr_cache (symbol, atr, timestamp) VALUES (?, ?, ?)",
            (symbol, atr, int(timestamp))
        )
        con.commit()
        con.close()
    except Exception as e:
        LOG.error("Failed to save ATR for %s to cache: %s", symbol, e)

def load_atr_from_cache() -> dict:
    """Loads all ATR values from the cache into a dictionary."""
    cache = {}
    if not CACHE_DB_PATH.exists():
        return cache
    try:
        con = sqlite3.connect(CACHE_DB_PATH)
        cur = con.cursor()
        for row in cur.execute("SELECT symbol, atr, timestamp FROM atr_cache"):
            cache[row[0]] = (row[1], row[2]) # (atr, timestamp)
        con.close()
        LOG.info("Loaded %d ATR values from cache.", len(cache))
    except Exception as e:
        LOG.error("Failed to load ATR from cache: %s", e)
    return cache