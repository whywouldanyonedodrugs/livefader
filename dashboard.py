# /opt/livefader/src/dashboard.py
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DataTable
from textual.containers import Container
from textual.timer import Timer

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
    def _db_symbol_to_ccxt(sym: str) -> str:
        """Convert DB symbol like 'MAGICUSDT' or 'BTCUSDT' to 'MAGIC/USDT'."""
        if "/" in sym:
            return sym
        if sym.endswith("USDT"):
            return f"{sym[:-4]}/USDT"
        if sym.endswith("USDC"):
            return f"{sym[:-4]}/USDC"
        return sym  # fall-back

    @staticmethod
    def _bar_chart(rows, k, v, width=30):
        if not rows:
            return "No data."
        mx = max(r[v] for r in rows if r[v] is not None)
        out = []
        for r in rows:
            label = str(r[k] or "N/A")
            val   = r[v] or 0
            bar   = "█" * int((val / mx) * width) if mx else ""
            out.append(f"{label:<15} | {bar} ({val})")
        return "\n".join(out)

    # (equity bar-chart unchanged; omitted for brevity)

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
        if not self.db_dsn:
            self.query_one("#kpi_pnl").update("DB_DSN not found")
            return

        self.pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)

        import ccxt.async_support as ccxt
        self.exchange = ccxt.bybit({"enableRateLimit": True})

        # only set titles here – NO add_columns
        self.query_one("#kpi_pnl").border_title          = "Total PnL"
        self.query_one("#kpi_win_rate").border_title      = "Win Rate"
        self.query_one("#kpi_profit_factor").border_title = "Profit Factor"
        self.query_one("#kpi_open_positions").border_title= "Open Positions"
        self.query_one("#equity_chart").border_title      = "Equity Curve"
        self.query_one("#regime_chart").border_title      = "Wins by Market Regime"
        self.query_one("#session_chart").border_title     = "Wins by Trading Session"
        self.query_one("#open_positions_table").border_title = "Live Positions"
        self.query_one("#recent_trades_table").border_title  = "Last 10 Closed Trades"

        self.set_interval(REFRESH_INTERVAL_SECONDS, self.update_data)
        await self.update_data()

    async def on_unmount(self) -> None:
        if self.exchange:
            await self.exchange.close()

    # ────────────────────────── main refresh ──────────────────────────
    async def update_data(self) -> None:
        if not self.pool:
            return
        try:
            kpi_q   = """SELECT
                COUNT(*) FILTER (WHERE status='OPEN')                       AS open_positions,
                SUM(pnl)   FILTER (WHERE status='CLOSED')                   AS total_pnl,
                COUNT(*) FILTER (WHERE status='CLOSED' AND pnl>0)           AS win_count,
                COUNT(*) FILTER (WHERE status='CLOSED')                     AS total_closed,
                SUM(pnl) FILTER (WHERE status='CLOSED' AND pnl>0)           AS gross_profit,
                SUM(pnl) FILTER (WHERE status='CLOSED' AND pnl<0)           AS gross_loss
            FROM positions"""
            open_q  = "SELECT symbol, side, size, entry_price FROM positions WHERE status='OPEN' ORDER BY opened_at DESC"
            recent_q= "SELECT symbol, pnl, exit_reason, holding_minutes FROM positions WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 10"
            equity_q= "SELECT equity FROM equity_snapshots ORDER BY ts ASC LIMIT 100"
            regime_q= "SELECT market_regime_at_entry AS regime, COUNT(*) AS wins FROM positions WHERE status='CLOSED' AND pnl>0 GROUP BY regime"
            sess_q  = "SELECT session_tag_at_entry   AS sess,   COUNT(*) AS wins FROM positions WHERE status='CLOSED' AND pnl>0 GROUP BY sess"

            kpis, open_pos, recent, eq_rows, reg_rows, sess_rows = await asyncio.gather(
                self.pool.fetchrow(kpi_q), self.pool.fetch(open_q),
                self.pool.fetch(recent_q), self.pool.fetch(equity_q),
                self.pool.fetch(regime_q), self.pool.fetch(sess_q),
            )
        except Exception as err:
            self.query_one("#kpi_pnl").update(f"ERROR DB:\n{err}")
            return

        # ── KPI boxes ──
        self.query_one("#kpi_open_positions").update(f"{kpis['open_positions'] or 0}")
        self.query_one("#kpi_pnl").update(f"${(kpis['total_pnl'] or 0):,.2f}")
        wins, closed = kpis["win_count"] or 0, kpis["total_closed"] or 0
        self.query_one("#kpi_win_rate").update(f"{(wins/closed*100 if closed else 0):.2f}%")
        gp, gl = kpis["gross_profit"] or 0, kpis["gross_loss"] or 0
        pf = gp / abs(gl) if gl else float("inf")
        self.query_one("#kpi_profit_factor").update(f"{pf:.2f}")

        # ── Equity curve (text barchart) ──
        eq_widget = self.query_one("#equity_chart")
        if len(eq_rows) > 1:
            curve = [float(r["equity"]) for r in eq_rows]
            start, curr = curve[0], curve[-1]
            pct = (curr / start - 1) * 100 if start else 0
            info = f" Start: ${start:,.2f}  Current: ${curr:,.2f} ({pct:+.2f}%)"
            eq_widget.update(f"{info}\n\n" + self._create_equity_barchart(curve))
        else:
            eq_widget.update("\nNot enough equity data yet.")

        # ── Regime / Session charts ──
        self.query_one("#regime_chart").update(self._bar_chart(reg_rows, "regime", "wins"))
        self.query_one("#session_chart").update(self._bar_chart(sess_rows,  "sess",   "wins"))

        # ── Live positions table ──
        open_tbl = self.query_one("#open_positions_table")
        open_tbl.clear(columns=True)  # wipe headers and rows
        open_tbl.add_columns("Symbol", "Side", "Size", "Entry Price", "Current Price", "UPnL ($)")

        # build CCXT keys
        ccxt_syms = [self._db_symbol_to_ccxt(p["symbol"]) for p in open_pos]
        try:
            tickers = await self.exchange.fetch_tickers(ccxt_syms) if ccxt_syms else {}
        except Exception:
            tickers = {}

        for p in open_pos:
            key  = self._db_symbol_to_ccxt(p["symbol"])
            last = tickers.get(key, {}).get("last", 0.0)
            entry = float(p["entry_price"]); size = float(p["size"])
            upnl  = (entry - last) * size if p["side"].lower() == "short" else (last - entry) * size

            open_tbl.add_row(
                p["symbol"], p["side"], f"{size}",
                f"{entry:.5f}", f"{last:.5f}", f"{upnl:+.2f}"
            )

        # ── Recent trades table ──
        recent_tbl = self.query_one("#recent_trades_table")
        recent_tbl.clear(columns=True)
        recent_tbl.add_columns("Symbol", "PnL", "Exit Reason", "Hold (m)")
        for r in recent:
            recent_tbl.add_row(
                r["symbol"], f"{r['pnl']:.2f}", r["exit_reason"], f"{r['holding_minutes']:.1f}"
            )

if __name__ == "__main__":
    DashboardApp().run()
