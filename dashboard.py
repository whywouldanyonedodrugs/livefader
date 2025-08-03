# /opt/livefader/src/dashboard.py

import asyncio
import asyncpg
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DataTable
from textual.containers import Container
from textual.timer import Timer

# --- Configuration ---
REFRESH_INTERVAL_SECONDS = 10 # How often to refresh the data

# This is our ASCII art logo. The 'r' before the string is important.
ASCII_LOGO = r"""
██╗     ██╗██╗   ██╗███████╗███████╗ █████╗ ██████╗ ███████╗██████╗ 
██║     ██║██║   ██║██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
██║     ██║██║   ██║█████╗  █████╗  ███████║██║  ██║█████╗  ██████╔╝
██║     ██║╚██╗ ██╔╝██╔══╝  ██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗
███████╗██║ ╚████╔╝ ███████╗██║     ██║  ██║██████╔╝███████╗██║  ██║
╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

class DashboardApp(App):
    """A retro, multi-panel TUI dashboard for the trading bot."""

    CSS = """
    Screen { color: #FFBF00; }
    #main_container { layout: grid; grid-size: 3; grid-gutter: 1; padding: 0 1; }
    #logo { column-span: 3; height: 9; content-align: center middle; }
    #kpi_container {
        /* FIX: Correct column span to use the full width */
        column-span: 3;
        layout: grid;
        grid-size: 4;
        grid-gutter: 1;
        height: 5;
    }
    .kpi_box, .chart_box {
        border: heavy #FFBF00;
        border-title-align: center;
        padding: 1;
    }
    /* FIX: Add explicit content alignment for KPI boxes */
    .kpi_box {
        content-align: center middle;
    }
    .chart_box { height: 12; }
    DataTable { column-span: 3; border: heavy #FFBF00; border-title-align: center; height: 12; }
    """

    TITLE = "LiveFader Trading Bot Dashboard"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.pool = None
        self.exchange = None
        load_dotenv()
        self.db_dsn = os.getenv("DATABASE_URL")

    def _create_bar_chart(self, data: list[dict], category_key: str, value_key: str, max_width: int = 30) -> str:
        if not data: return "No data available."
        max_val = max(item[value_key] for item in data if item[value_key] is not None) if data else 0
        chart = []
        for item in data:
            category = str(item[category_key]) if item[category_key] else "N/A"
            value = item[value_key] or 0
            bar_len = int((value / max_val) * max_width) if max_val > 0 else 0
            bar = "█" * bar_len
            chart.append(f"{category:<15} | {bar} ({value})")
        return "\n".join(chart)

    def _create_equity_barchart(self, equity_data: list[float], num_bars: int = 10, height: int = 8) -> str:
        if len(equity_data) < 2: return "Not enough data."
        min_eq, max_eq = min(equity_data), max(equity_data)
        if max_eq == min_eq: return "Equity is flat."
        
        bin_size = len(equity_data) // num_bars if num_bars > 0 else 1
        bins = [equity_data[i:i + bin_size][-1] for i in range(0, len(equity_data), bin_size)][:num_bars]
        
        normalized = [((val - min_eq) / (max_eq - min_eq)) * (height - 1) for val in bins]
        
        chart = [""] * height
        for h in range(height - 1, -1, -1):
            row = ""
            for val in normalized:
                row += "███ " if val >= h else "    "
            chart[height - 1 - h] = row
        
        return "\n".join(chart)

    def compose(self) -> ComposeResult:
        # ... (This function is correct and does not need changes) ...
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

    async def on_mount(self) -> None:
        # ... (This function is correct and does not need changes) ...
        if not self.db_dsn:
            self.query_one("#kpi_pnl").border_title = "Error"
            self.query_one("#kpi_pnl").update("DB_DSN not found")
            return
        
        self.pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)
        
        import ccxt.async_support as ccxt
        self.exchange = ccxt.bybit({'enableRateLimit': True})

        self.query_one("#kpi_pnl").border_title = "Total PnL"
        self.query_one("#kpi_win_rate").border_title = "Win Rate"
        self.query_one("#kpi_profit_factor").border_title = "Profit Factor"
        self.query_one("#kpi_open_positions").border_title = "Open Positions"
        self.query_one("#equity_chart").border_title = "Equity Curve"
        self.query_one("#regime_chart").border_title = "Wins by Market Regime"
        self.query_one("#session_chart").border_title = "Wins by Trading Session"

        open_pos_table = self.query_one("#open_positions_table")
        open_pos_table.border_title = "Live Positions"
        open_pos_table.add_columns("Symbol", "Side", "Size", "Entry Price", "Current Price", "UPnL ($)")
        
        recent_trades_table = self.query_one("#recent_trades_table")
        recent_trades_table.border_title = "Last 10 Closed Trades"
        recent_trades_table.add_columns("Symbol", "PnL", "Exit Reason", "Hold (m)")

        self.update_timer: Timer = self.set_interval(REFRESH_INTERVAL_SECONDS, self.update_data)
        await self.update_data()

    async def on_unmount(self) -> None:
        if self.exchange: await self.exchange.close()

    async def update_data(self) -> None:
        """Fetch fresh data from the database and update widgets."""
        if not self.pool:
            return

        try:
            kpi_query = """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'OPEN') AS open_positions,
                    SUM(pnl) FILTER (WHERE status = 'CLOSED') AS total_pnl,
                    COUNT(*) FILTER (WHERE status = 'CLOSED' AND pnl > 0) AS win_count,
                    COUNT(*) FILTER (WHERE status = 'CLOSED') AS total_closed,
                    SUM(pnl) FILTER (WHERE status = 'CLOSED' AND pnl > 0) AS gross_profit,
                    SUM(pnl) FILTER (WHERE status = 'CLOSED' AND pnl < 0) AS gross_loss
                FROM positions
            """
            open_pos_query = "SELECT symbol, side, size, entry_price FROM positions WHERE status = 'OPEN' ORDER BY opened_at DESC"
            recent_trades_query = "SELECT symbol, pnl, exit_reason, holding_minutes FROM positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT 10"
            equity_query = "SELECT equity FROM equity_snapshots ORDER BY ts ASC LIMIT 100"
            regime_query = "SELECT market_regime_at_entry, COUNT(*) as wins FROM positions WHERE status = 'CLOSED' AND pnl > 0 GROUP BY market_regime_at_entry"
            session_query = "SELECT session_tag_at_entry, COUNT(*) as wins FROM positions WHERE status = 'CLOSED' AND pnl > 0 GROUP BY session_tag_at_entry"

            kpis, open_positions, recent_trades, equity_records, regime_wins, session_wins = await asyncio.gather(
                self.pool.fetchrow(kpi_query), self.pool.fetch(open_pos_query),
                self.pool.fetch(recent_trades_query), self.pool.fetch(equity_query),
                self.pool.fetch(regime_query), self.pool.fetch(session_query)
            )

            # --- Update KPI Widgets ---
            self.query_one("#kpi_open_positions").update(f"{kpis['open_positions'] or 0}")
            total_pnl = kpis['total_pnl'] or 0.0
            self.query_one("#kpi_pnl").update(f"${total_pnl:,.2f}")
            win_count = kpis['win_count'] or 0
            total_closed = kpis['total_closed'] or 0
            win_rate = (win_count / total_closed) * 100 if total_closed > 0 else 0.0
            self.query_one("#kpi_win_rate").update(f"{win_rate:.2f}%")
            gross_profit = kpis['gross_profit'] or 0.0
            gross_loss = kpis['gross_loss'] or 0.0
            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
            self.query_one("#kpi_profit_factor").update(f"{profit_factor:.2f}")

            # --- Update Text-Based Equity Curve ---
            equity_chart = self.query_one("#equity_chart")
            if equity_records and len(equity_records) > 1:
                equity_data = [float(r['equity']) for r in equity_records]
                start_eq, current_eq = equity_data[0], equity_data[-1]
                min_eq, max_eq = min(equity_data), max(equity_data)
                change_pct = (current_eq / start_eq - 1) * 100 if start_eq > 0 else 0
                
                info_line = f" Start: ${start_eq:,.2f}  Current: ${current_eq:,.2f} ({change_pct:+.2f}%)"
                chart = self._create_equity_barchart(equity_data)
                equity_chart.update(f"{info_line}\n\n{chart}")
            else:
                equity_chart.update("\nNot enough equity data yet.")

            # --- Update Text-Based Bar Charts ---
            self.query_one("#regime_chart").update(self._create_bar_chart(regime_wins, 'market_regime_at_entry', 'wins'))
            self.query_one("#session_chart").update(self._create_bar_chart(session_wins, 'session_tag_at_entry', 'wins'))

            # --- Update Live Positions Table ---
            open_pos_table = self.query_one("#open_positions_table")
            open_pos_table.clear()
            if open_positions:
                # Create the list of symbols in the format ccxt expects (e.g., 'MUSDT/USDT')
                symbols_to_fetch = [f"{pos['symbol']}/USDT" for pos in open_positions]
                tickers = await self.exchange.fetch_tickers(symbols_to_fetch)
                
                for pos in open_positions:
                    symbol_db = pos['symbol']
                    # FIX: Use the correctly formatted key for the lookup
                    ticker_key = f"{symbol_db}/USDT"
                    
                    current_price = tickers.get(ticker_key, {}).get('last', 0.0)
                    entry_price = float(pos['entry_price'])
                    size = float(pos['size'])
                    upnl = (entry_price - current_price) * size if pos['side'] == 'short' else (current_price - entry_price) * size
                    
                    open_pos_table.add_row(
                        symbol_db, pos['side'], f"{size}", f"{entry_price:.5f}",
                        f"{current_price:.5f}", f"{upnl:+.2f}"
                    )

            # --- Update Recent Trades Table ---
            recent_trades_table = self.query_one("#recent_trades_table")
            recent_trades_table.clear()
            if recent_trades:
                for trade in recent_trades:
                    recent_trades_table.add_row(trade['symbol'], f"{trade['pnl']:.2f}", trade['exit_reason'], f"{trade['holding_minutes']:.1f}")

        except Exception as e:
            self.query_one("#kpi_pnl").update(f"ERROR:\n{e}")

if __name__ == "__main__":
    app = DashboardApp()
    app.run()