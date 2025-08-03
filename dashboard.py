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
    """A retro, amber-themed TUI dashboard for the trading bot."""

    # --- CSS for the "oldschool" layout and amber theme ---
    CSS = """
    Screen {
        /* Define our amber color variable */
        $amber: #FFBF00;
        /* Set all text and borders to amber */
        color: $amber;
        border-color: $amber;
    }
    #main_container {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 0 1;
    }
    #logo {
        column-span: 2;
        height: 9;
        content-align: center middle;
    }
    #kpi_container {
        column-span: 2;
        layout: grid;
        grid-size: 4;
        grid-gutter: 1;
        height: 5;
    }
    .kpi_box {
        /* Use a heavy, solid border */
        border: heavy $amber;
        border-title-align: center;
    }
    .kpi_value {
        text-align: center;
        padding-top: 1;
    }
    DataTable {
        /* Use a heavy, solid border with a title */
        border: heavy $amber;
        border-title-align: center;
        height: 12;
    }
    """

    TITLE = "LiveFader Trading Bot Dashboard"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.pool = None
        load_dotenv()
        self.db_dsn = os.getenv("DATABASE_URL")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        with Container(id="main_container"):
            yield Static(ASCII_LOGO, id="logo")
            with Container(id="kpi_container"):
                yield Static("", classes="kpi_box", id="kpi_pnl")
                yield Static("", classes="kpi_box", id="kpi_win_rate")
                yield Static("", classes="kpi_box", id="kpi_profit_factor")
                yield Static("", classes="kpi_box", id="kpi_open_positions")
            
            yield DataTable(id="open_positions_table")
            yield DataTable(id="recent_trades_table")
        yield Footer()

    async def on_mount(self) -> None:
        """Called when the app is first mounted."""
        if not self.db_dsn:
            self.query_one("#kpi_pnl").border_title = "Error"
            self.query_one("#kpi_pnl").update("DB_DSN not found")
            return
        
        self.pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)
        
        # Set titles for our widgets
        self.query_one("#kpi_pnl").border_title = "Total PnL"
        self.query_one("#kpi_win_rate").border_title = "Win Rate"
        self.query_one("#kpi_profit_factor").border_title = "Profit Factor"
        self.query_one("#kpi_open_positions").border_title = "Open Positions"

        open_pos_table = self.query_one("#open_positions_table")
        open_pos_table.border_title = "Live Positions"
        open_pos_table.add_columns("Symbol", "Side", "Size", "Entry Price")
        
        recent_trades_table = self.query_one("#recent_trades_table")
        recent_trades_table.border_title = "Last 5 Closed Trades"
        recent_trades_table.add_columns("Symbol", "PnL", "Exit Reason", "Hold (m)")

        self.update_timer: Timer = self.set_interval(REFRESH_INTERVAL_SECONDS, self.update_data)
        await self.update_data()

    async def update_data(self) -> None:
        """Fetch fresh data from the database and update widgets."""
        if not self.pool:
            return

        try:
            kpi_query = """
                SELECT
                    (SELECT COUNT(*) FROM positions WHERE status = 'OPEN') as open_positions,
                    (SELECT SUM(pnl) FROM positions WHERE status = 'CLOSED') as total_pnl,
                    (SELECT COUNT(*) FROM positions WHERE status = 'CLOSED' AND pnl > 0) as win_count,
                    (SELECT COUNT(*) FROM positions WHERE status = 'CLOSED') as total_closed,
                    (SELECT SUM(pnl) FROM positions WHERE status = 'CLOSED' AND pnl > 0) as gross_profit,
                    (SELECT SUM(pnl) FROM positions WHERE status = 'CLOSED' AND pnl < 0) as gross_loss
            """
            kpis = await self.pool.fetchrow(kpi_query)

            open_positions = await self.pool.fetch("SELECT symbol, side, size, entry_price FROM positions WHERE status = 'OPEN' ORDER BY opened_at DESC")
            recent_trades = await self.pool.fetch("SELECT symbol, pnl, exit_reason, holding_minutes FROM positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT 5")

            # --- Update KPI Widgets (without color markup) ---
            self.query_one("#kpi_open_positions").update(f"\n{kpis['open_positions']}")
            
            total_pnl = kpis['total_pnl'] or 0
            self.query_one("#kpi_pnl").update(f"\n${total_pnl:,.2f}")
            
            win_rate = (kpis['win_count'] / kpis['total_closed']) * 100 if kpis['total_closed'] > 0 else 0
            self.query_one("#kpi_win_rate").update(f"\n{win_rate:.2f}%")
            
            profit_factor = kpis['gross_profit'] / abs(kpis['gross_loss']) if kpis['gross_loss'] and kpis['gross_loss'] != 0 else float('inf')
            self.query_one("#kpi_profit_factor").update(f"\n{profit_factor:.2f}")

            # --- Update Tables ---
            open_pos_table = self.query_one("#open_positions_table")
            open_pos_table.clear()
            for pos in open_positions:
                open_pos_table.add_row(pos['symbol'], pos['side'], f"{pos['size']}", f"{pos['entry_price']}")

            recent_trades_table = self.query_one("#recent_trades_table")
            recent_trades_table.clear()
            for trade in recent_trades:
                recent_trades_table.add_row(trade['symbol'], f"{trade['pnl']:.2f}", trade['exit_reason'], f"{trade['holding_minutes']:.1f}")

        except Exception as e:
            self.query_one("#kpi_pnl").update(f"ERROR:\n{e}")

if __name__ == "__main__":
    app = DashboardApp()
    app.run()