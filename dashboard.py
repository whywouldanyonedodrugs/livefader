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
        max_bars: int = 30,
        min_height: int = 5,           # 👈 at least this tall
    ) -> str:
        """
        Classic OHLC bar (open tick left, close tick right).
        One bar = 3 text columns, colour-coded:
        • bright_green = close ≥ open
        • bright_red   = close <  open
        """

        if len(ohlcv) < 2:
            return "Not enough data."

        # — 1) down-sample so we never exceed max_bars —
        step  = max(1, len(ohlcv) // max_bars)
        data  = ohlcv[-step * max_bars :: step]

        # allocate canvas (rows × 3·bars)
        W = 3 * len(data)
        grid = [[" "] * W for _ in range(rows)]

        # helper: centre one bar inside the full height
        def y_map(high: float, low: float, price: float) -> int:
            span = max(high - low, 1e-9)
            # force stem to be ≥ min_height
            span = max(span, 1 / rows * min_height * span)
            mid  = (high + low) / 2
            half = span / 2
            offset = (price - mid) / half
            return int(rows // 2 - offset * (rows // 2 - 1))

        for idx, (_, o, h, l, c, _) in enumerate(data):
            up   = c >= o
            clr  = "bright_green" if up else "bright_red"
            xmid = 3 * idx + 1           # centre column of this bar

            # map prices → rows
            top = y_map(h, l, h)
            bot = y_map(h, l, l)
            y_open  = y_map(h, l, o)
            y_close = y_map(h, l, c)

            # draw stem first
            for r in range(min(top, bot), max(top, bot) + 1):
                grid[r][xmid] = f"[{clr}]│[/]"

            # overwrite with ticks so they sit on top of the stem
            grid[y_open ][xmid - 1] = f"[{clr}]─[/]"
            grid[y_close][xmid + 1] = f"[{clr}]─[/]"

        return "\n".join("".join(r) for r in grid)

    
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
        panel.border_title = f"{pair} – {len(ohlcv)} × 15 m (OHLC)"
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
