# /opt/livefader/src/dashboard.py
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DataTable
from textual.containers import Container
from textual.timer import Timer
from rich.text import Text
from textual.message import Message
from textual.events import Click
from textual import on            # ← add this


REFRESH_INTERVAL_SECONDS = 10  # dashboard update cadence

ASCII_LOGO = r"""
██╗     ██╗██╗   ██╗███████╗███████╗ █████╗ ██████╗ ███████╗██████╗ 
██║     ██║██║   ██║██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
██║     ██║██║   ██║█████╗  █████╗  ███████║██║  ██║█████╗  ██████╔╝
██║     ██║╚██╗ ██╔╝██╔══╝  ██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗
███████╗██║ ╚████╔╝ ███████╗██║     ██║  ██║██████╔╝███████╗██║  ██║
╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

class DashboardApp(App):
    CSS = """
    Screen { color: #FFBF00; }
    #main_container { layout: grid; grid-size: 3; grid-gutter: 1; padding: 0 1; }
    #logo { column-span: 3; height: 9; content-align: center middle; }
    #kpi_container { column-span: 3; layout: grid; grid-size: 4; grid-gutter: 1; height: 5; }
    .kpi_box, .chart_box { border: heavy #FFBF00; border-title-align: center; padding: 1; }
    .kpi_box { content-align: center middle; }
    .chart_box { height: 12; }
    DataTable { column-span: 3; border: heavy #FFBF00; border-title-align: center; height: 12; }
    """
    TITLE = "LiveFader Trading Bot Dashboard"
    BINDINGS = [("q", "quit", "Quit")]

    # ────────────────────────── helpers ──────────────────────────

    @staticmethod
    def _resolve_perp_symbol(exchange, raw: str) -> str:
        """
        Return a CCXT-recognised symbol for a USDT-perp.

        • raw can be 'BTCUSDT', '1000TOSHIUSDT', 'BTC/USDT', etc.
        • Result will be something like 'BTC/USDT:USDT' or unchanged if already OK.
        """
        raw = raw.upper().replace("/", "")
        base = raw.replace("USDT", "")
        # Most Bybit USDT-perps are shaped like 'BASE/USDT:USDT'
        perp = f"{base}/USDT:USDT"
        if perp in exchange.markets:
            return perp
        # Fallback to the raw value if CCXT already knows it
        if raw in exchange.markets:
            return exchange.markets[raw]["symbol"]
        # Last resort: return the spot pair so we at least draw a chart
        return f"{base}/USDT"

    @staticmethod
    def _ascii_ohlc_bars(
        ohlcv: list[list[float]],
        rows: int = 15,
        max_bars: int = 30,        # number of OHLC bars, not text columns
        centre_h: bool = True,
        centre_v: bool = True,
    ) -> str:
        """
        Return coloured OHLC bars (open tick left, close tick right).

        • Up-bar  (close ≥ open) → bright_green
        • Down-bar                → bright_red
        """
        if len(ohlcv) < 2:
            return "Not enough data."

        # ▸ down-sample so we never draw more than max_bars bars
        stride = max(1, len(ohlcv) // max_bars)
        data   = ohlcv[-stride * max_bars :: stride]

        hi = max(r[2] for r in data)
        lo = min(r[3] for r in data)
        span = (hi - lo) or 1e-9
        compress = 0.4                      # 0 – 1  (smaller ⇒ “squash” bars)
        usable   = max(1, int(rows * compress))
        pad_top  = (rows - usable) // 2       # equal top / bottom padding
        pad_bot  = rows - usable - pad_top
        rows_minus1 = usable - 1

        # --- vertical mapping: centre on mid-price --------------------------
        mid   = (hi + lo) / 2
        half  = max(hi - mid, mid - lo) or 1e-9        # symmetric span
        center_row  = rows // 2                        # zero-based
        scale_rows  = rows // 2 - 1                    # rows above / below mid

        def y(price: float) -> int:
            """
            Map price → row index such that
            mid  → center_row
            mid+half → top of band
            mid-half → bottom of band
            """
            offset = (price - mid) / half
            return int(center_row - offset * scale_rows)

        for i, (_, o, h, l, c, _) in enumerate(data):
            col  = "bright_green" if c >= o else "bright_red"
            x0   = 3 * i + 1                        # centre column for the stem
            top, bot      = y(h), y(l)
            y_open, y_close = y(o), y(c)

            # stem (vertical line)
            for r in range(bot, top + 1):
                canvas[rows_minus1 - r][x0] = f"[{col}]│[/]"

            # open tick (left)
            canvas[rows_minus1 - y_open][x0 - 1] = f"[{col}]─[/]"

            # close tick (right)
            canvas[rows_minus1 - y_close][x0 + 1] = f"[{col}]─[/]"

        # join rows
        lines = ["".join(r) for r in canvas]

        # horizontal centring
        if centre_h and len(data) < max_bars:
            pad_cols = (max_bars - len(data)) // 2 * 3
            pad = " " * pad_cols
            lines = [pad + ln + pad for ln in lines]

        # vertical centring
        if centre_v:
            non_blank = [i for i, ln in enumerate(lines) if ln.strip()]
            if non_blank:
                top, bot = min(non_blank), max(non_blank)
                used = bot - top + 1
                pad_top  = (rows - used) // 2
                pad_bot  = rows - used - pad_top
                blank    = " " * len(lines[0])
                lines = [blank]*pad_top + lines[top:bot+1] + [blank]*pad_bot

        return "\n".join(lines)

    
    @staticmethod
    def _db_sym_to_pair(symbol: str) -> str:
        """
        Convert symbols stored like 'BTCUSDT', 'MAGICUSDT', 'MUSDT'
        into CCXT style 'BTC/USDT', 'MAGIC/USDT', 'M/USDT'.
        """
        symbol = symbol.upper()
        if "/" in symbol:                # already a pair
            return symbol
        for quote in ("USDT", "USDC", "BTC", "ETH"):
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]
                return f"{base}/{quote}"
        return symbol                    # fallback (unlikely)

    @staticmethod
    def _bar_chart(rows, key, val_key, width=30):
        if not rows:
            return "No data."
        m = max(r[val_key] for r in rows if r[val_key] is not None)
        lines = []
        for r in rows:
            label = str(r[key] or "N/A")
            value = r[val_key] or 0
            bar_len = int((value / m) * width) if m else 0
            lines.append(f"{label:<15} | {'█'*bar_len} ({value})")
        return "\n".join(lines)

    @staticmethod
    def _equity_barchart(curve, bars=10, height=8):
        if len(curve) < 2:
            return "Not enough data."
        mn, mx = min(curve), max(curve)
        if mx == mn:
            return "Equity is flat."
        step   = max(1, len(curve) // bars)
        samples = [curve[i] for i in range(step-1, len(curve), step)][:bars]
        scaled  = [((v-mn)/(mx-mn))*(height-1) for v in samples]
        rows = []
        for h in reversed(range(height)):
            row = "".join("███ " if v >= h else "    " for v in scaled)
            rows.append(row)
        return "\n".join(rows)

    # ── tiny helpers to avoid duplicate headers ─────────────────────────
    @staticmethod
    def _ensure_live_headers(table: DataTable) -> None:
        """Add the six live-position columns once, if they’re missing."""
        if not table.columns:                     # <— works on every Textual
            table.add_columns(
                "Symbol", "Side", "Size",
                "Entry Price", "Current Price", "UPnL ($)"
            )

    @staticmethod
    def _ensure_trade_headers(table: DataTable) -> None:
        """Add the four recent-trade columns once, if they’re missing."""
        if not table.columns:
            table.add_columns("Symbol", "PnL", "Exit Reason", "Hold (m)")

    @on(DataTable.RowHighlighted)          # or RowSelected
    async def on_data_table_row_highlighted(
        self, message: DataTable.RowHighlighted
    ) -> None:
        table = message.control
        row_i = message.cursor_row
        try:
            raw_sym = table.get_cell_at((row_i, 0))   # first column = Symbol
        except Exception:
            return                                    # header / empty click

        # build a valid Bybit symbol (perp or spot) and fetch 15-minute candles
        await self.exchange.load_markets()
        pair = self._resolve_perp_symbol(self.exchange, raw_sym)

        try:
            ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe="30m", limit=8)
        except Exception as e:
            self.query_one("#equity_chart").update(f"Failed to fetch OHLCV: {e}")
            return

                                        # header / empty click

        chart  = DashboardApp._ascii_ohlc_bars(ohlcv, rows=15, max_bars=30)
        panel  = self.query_one("#candle_chart")
        panel.border_title = f"{pair} – {len(ohlcv)} × 15 m  (OHLC)"
        panel.update(Text.from_markup(chart))

    # ────────────────────────── compose ──────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main_container"):
            yield Static(ASCII_LOGO, id="logo")
            with Container(id="kpi_container"):
                yield Static("", classes="kpi_box", id="kpi_pnl")
                yield Static("", classes="kpi_box", id="kpi_win_rate")
                yield Static("", classes="kpi_box", id="kpi_profit_factor")
                yield Static("", classes="kpi_box", id="kpi_open_positions")
            yield Static("", classes="chart_box", id="equity_chart")
            yield Static("", classes="chart_box", id="regime_chart")
            yield Static("", classes="chart_box", id="session_chart")
            yield Static("", classes="chart_box", id="candle_chart")            
            yield DataTable(id="open_positions_table")
            yield DataTable(id="recent_trades_table")
        yield Footer()

    # ────────────────────────── life-cycle ──────────────────────────
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.db_dsn = os.getenv("DATABASE_URL")
        self.pool = None
        self.exchange = None

    async def on_mount(self) -> None:
        self.query_one("#candle_chart").border_title = "15-min Preview"

        if not self.db_dsn:
            self.query_one("#kpi_pnl").update("DB_DSN not set")
            return

        self.pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)

        import ccxt.async_support as ccxt
        self.exchange = ccxt.bybit({"enableRateLimit": True})

        # set the border titles (no add_columns here!)
        self.query_one("#kpi_pnl").border_title            = "Total PnL"
        self.query_one("#kpi_win_rate").border_title        = "Win Rate"
        self.query_one("#kpi_profit_factor").border_title   = "Profit Factor"
        self.query_one("#kpi_open_positions").border_title  = "Open Positions"
        self.query_one("#equity_chart").border_title        = "Equity Curve"
        self.query_one("#regime_chart").border_title        = "Wins by Market Regime"
        self.query_one("#session_chart").border_title       = "Wins by Trading Session"
        self.query_one("#open_positions_table").border_title= "Live Positions"
        self.query_one("#recent_trades_table").border_title = "Last 10 Closed Trades"

        self.set_interval(REFRESH_INTERVAL_SECONDS, self.update_data)
        await self.update_data()

        open_tbl = self.query_one("#open_positions_table")
        open_tbl.cursor_type  = "row"
        open_tbl.show_cursor  = True
        open_tbl.focus()   

    async def on_unmount(self) -> None:
        if self.exchange:
            await self.exchange.close()

    # ────────────────────────── main refresh ──────────────────────────
    async def update_data(self) -> None:
        if not self.pool:
            return

        # ── fetch everything from DB ──
        try:
            kpi_q = """
                SELECT
                    COUNT(*) FILTER (WHERE status='OPEN')                    AS open_positions,
                    SUM(pnl)   FILTER (WHERE status='CLOSED')                AS total_pnl,
                    COUNT(*) FILTER (WHERE status='CLOSED' AND pnl>0)        AS wins,
                    COUNT(*) FILTER (WHERE status='CLOSED')                  AS closed,
                    SUM(pnl) FILTER (WHERE status='CLOSED' AND pnl>0)        AS g_profit,
                    SUM(pnl) FILTER (WHERE status='CLOSED' AND pnl<0)        AS g_loss
                FROM positions
            """
            open_q   = "SELECT symbol, side, size, entry_price FROM positions WHERE status='OPEN' ORDER BY opened_at DESC"
            recent_q = "SELECT symbol, pnl, exit_reason, holding_minutes FROM positions WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 10"
            equity_q = "SELECT equity FROM equity_snapshots ORDER BY ts ASC LIMIT 100"
            regime_q = "SELECT market_regime_at_entry AS regime, COUNT(*) AS wins FROM positions WHERE status='CLOSED' AND pnl>0 GROUP BY regime"
            sess_q   = "SELECT session_tag_at_entry   AS sess,   COUNT(*) AS wins FROM positions WHERE status='CLOSED' AND pnl>0 GROUP BY sess"

            kpis, open_pos, recent, equity_rows, reg_rows, sess_rows = await asyncio.gather(
                self.pool.fetchrow(kpi_q), self.pool.fetch(open_q),
                self.pool.fetch(recent_q), self.pool.fetch(equity_q),
                self.pool.fetch(regime_q), self.pool.fetch(sess_q)
            )
        except Exception as err:
            self.query_one("#kpi_pnl").update(f"ERROR DB:\n{err}")
            return

        # ── KPI boxes ──
        self.query_one("#kpi_open_positions").update(str(kpis["open_positions"] or 0))
        self.query_one("#kpi_pnl").update(f"${(kpis['total_pnl'] or 0):,.2f}")
        wins, closed = kpis["wins"] or 0, kpis["closed"] or 0
        self.query_one("#kpi_win_rate").update(f"{(wins/closed*100 if closed else 0):.2f}%")
        gp, gl = kpis["g_profit"] or 0, kpis["g_loss"] or 0
        pf = gp / abs(gl) if gl else float("inf")
        self.query_one("#kpi_profit_factor").update(f"{pf:.2f}")

        # ── Equity curve widget ──
        eq_widget = self.query_one("#equity_chart")
        if len(equity_rows) > 1:
            curve = [float(r["equity"]) for r in equity_rows]
            start, curr = curve[0], curve[-1]
            pct = (curr / start - 1) * 100 if start else 0
            info = f" Start: ${start:,.2f}  Current: ${curr:,.2f} ({pct:+.2f}%)"
            eq_widget.update(info + "\n\n" + self._equity_barchart(curve))
        else:
            eq_widget.update("\nNot enough equity data yet.")

        # ── regime / session bar charts ──
        self.query_one("#regime_chart").update(self._bar_chart(reg_rows,  "regime", "wins"))
        self.query_one("#session_chart").update(self._bar_chart(sess_rows, "sess",  "wins"))

        # ── Live positions table ──
        open_tbl = self.query_one("#open_positions_table")
        open_tbl.clear()                     # wipe rows *and* headers
        self._ensure_live_headers(open_tbl)  # ← add headers once

        # For each open position, pull a ticker individually and compute uPNL
        for pos in open_pos:
            # build a list of symbol strings to try with Bybit:
            #   1) the perpetual form (BTCUSDT)
            #   2) the spot-style form (BTC/USDT)
            #   3) whatever string was stored in the DB
            base  = pos["symbol"].replace("USDT", "").replace("/", "")
            attempts = [
                f"{base}USDT",          # Bybit perp
                f"{base}/USDT",         # generic spot
                pos["symbol"],          # stored value
            ]

            last_price = 0.0
            for sym in attempts:
                try:
                    tick = await self.exchange.fetch_ticker(sym)
                    last_price = tick.get("last") or 0.0
                except Exception:
                    last_price = 0.0
                if last_price:
                    break   # stop at the first successful lookup

            entry_price = float(pos["entry_price"])
            size = float(pos["size"])
            if pos["side"].lower() == "short":
                upnl = (entry_price - last_price) * size
            else:
                upnl = (last_price - entry_price) * size

            open_tbl.add_row(
                pos["symbol"],
                pos["side"],
                f"{size}",
                f"{entry_price:.5f}",
                f"{last_price:.5f}",
                f"{upnl:+.2f}",
            )

        # ── Recent trades table ──
        recent_tbl = self.query_one("#recent_trades_table")
        recent_tbl.clear()
        self._ensure_trade_headers(recent_tbl)  
        
        for r in recent:
            recent_tbl.add_row(
                r["symbol"],
                f"{r['pnl']:.2f}",
                r["exit_reason"],
                f"{r['holding_minutes']:.1f}",
            ) # ← add headers once
            

if __name__ == "__main__":
    DashboardApp().run()
