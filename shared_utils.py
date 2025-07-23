# shared_utils.py
"""
Shared utility functions for data loading and configuration management.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Dict, List

import pandas as pd
import pyarrow.parquet as pq

import config as cfg

# --- Cache for frequently accessed data ---
_blacklist_cache: Set[str] | None = None
_symbol_map_cache: Dict[str, str] | None = None
_cg_details_cache: Dict[str, Dict] | None = None


def get_symbols_from_file(symbols_file: Path = cfg.SYMBOLS_FILE) -> List[str]:
    """Parses the symbols.txt file and returns a list of uppercase symbols."""
    if not symbols_file.exists():
        raise FileNotFoundError(f"{symbols_file} not found – place it in the project root.")

    with symbols_file.open() as fh:
        symbols = [
            line.split("#", 1)[0].strip().upper()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not symbols:
        raise ValueError(f"{symbols_file} is empty – nothing to process.")
    return symbols


def load_parquet_data(symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Loads a single symbol's data from a Parquet file with optional date filtering."""
    pq_path = cfg.PARQUET_DIR / f"{symbol}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet file for {symbol} is missing. Run etl.py first.")

    df = pq.read_table(pq_path).to_pandas()

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df


def load_blacklist_data() -> None:
    """Loads CoinGecko blacklist, symbol map, and details cache into memory."""
    global _blacklist_cache, _symbol_map_cache, _cg_details_cache

    if _blacklist_cache is None:
        bl_path = cfg.PROJECT_ROOT / "blacklist.txt"
        if bl_path.exists():
            with bl_path.open() as f:
                _blacklist_cache = {line.strip().lower() for line in f if line.strip()}
        else:
            _blacklist_cache = set()

    if _symbol_map_cache is None:
        map_file = cfg.PROJECT_ROOT / "coingecko_map.json"
        if map_file.exists():
            with map_file.open() as f:
                _symbol_map_cache = {c["symbol"].upper(): c["id"] for c in json.load(f)}
        else:
            _symbol_map_cache = {}

    if _cg_details_cache is None:
        det_file = cfg.PROJECT_ROOT / "coingecko_details_cache.json"
        if det_file.exists():
            with det_file.open() as f:
                _cg_details_cache = json.load(f)
        else:
            _cg_details_cache = {}


def is_blacklisted(symbol: str) -> bool:
    """Checks if a symbol belongs to a blacklisted CoinGecko category."""
    if _blacklist_cache is None:
        load_blacklist_data()

    if not _blacklist_cache or not _symbol_map_cache:
        return False

    base_symbol = symbol.replace("USDT", "").upper()
    cg_id = _symbol_map_cache.get(base_symbol)
    if not cg_id:
        return False

    categories = _cg_details_cache.get(cg_id, {}).get("categories", [])
    return any(cat and cat.lower() in _blacklist_cache for cat in categories)