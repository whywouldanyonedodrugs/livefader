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

class DashboardApp(App):
    """A live-updating Text User Interface (TUI) for the trading bot."""

    # CSS for layout and styling
    CSS = """
    #main_container {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }
    #kpi_container {
        column-span: 2;
        layout: grid;
        grid-size: 4;
        grid-gutter: 1;
        height: 5;
    }
    .kpi_box {
        border: round white;
    }
    .kpi_title {
        text-align: center;
        background: $primary-background-darken-2;
    }
    .kpi_value {
        text-align: center;
        font-size: 120%;
        padding-top: 1;
    }
    #open_positions_table {
        border: round white;
        height: 12;
    }
    #recent_trades_table {
        border: round white;
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
        yield Header()
        with Container(id="main_container"):
            with Container(id="kpi_container"):
                yield Static("[b]Total PnL[/b]\n\n--", classes="kpi_box", id="kpi_pnl")
                yield Static("[b]Win Rate[/b]\n\n--", classes="kpi_box", id="kpi_win_rate")
                yield Static("[b]Profit Factor[/b]\n\n--", classes="kpi_box", id="kpi_profit_factor")
                yield Static("[b]Open Positions[/b]\n\n--", classes="kpi_box", id="kpi_open_positions")
            
            yield DataTable(id="open_positions_table")
            yield DataTable(id="recent_trades_table")
        yield Footer()

    async def on_mount(self) -> None:
        """Called when the app is first mounted."""
        if not self.db_dsn:
            self.query_one("#kpi_pnl").update("[b]Total PnL[/b]\n\n[red]ERROR: DB_DSN[/red]")
            return
        
        self.pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)
        
        # Set up tables
        open_pos_table = self.query_one("#open_positions_table")
        open_pos_table.add_columns("Symbol", "Side", "Size", "Entry Price", "PnL (Unrealized)")
        
        recent_trades_table = self.query_one("#recent_trades_table")
        recent_trades_table.add_columns("Symbol", "Closed At", "PnL", "Exit Reason", "Hold (m)")

        # Start the update timer
        self.update_timer: Timer = self.set_interval(REFRESH_INTERVAL_SECONDS, self.update_data)
        await self.update_data() # Run once immediately

    async def update_data(self) -> None:
        """Fetch fresh data from the database and update widgets."""
        if not self.pool:
            return

        try:
            # --- Fetch KPI Data ---
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

            # --- Fetch Open Positions ---
            open_pos_query = "SELECT symbol, side, size, entry_price, pnl FROM positions WHERE status = 'OPEN' ORDER BY opened_at DESC"
            open_positions = await self.pool.fetch(open_pos_query)

            # --- Fetch Recent Trades ---
            recent_trades_query = "SELECT symbol, closed_at, pnl, exit_reason, holding_minutes FROM positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT 5"
            recent_trades = await self.pool.fetch(recent_trades_query)

            # --- Update KPI Widgets ---
            self.query_one("#kpi_open_positions").update(f"[b]Open Positions[/b]\n\n{kpis['open_positions']}")
            self.query_one("#kpi_pnl").update(f"[b]Total PnL[/b]\n\n${kpis['total_pnl'] or 0:,.2f}")
            
            win_rate = (kpis['win_count'] / kpis['total_closed']) * 100 if kpis['total_closed'] > 0 else 0
            self.query_one("#kpi_win_rate").update(f"[b]Win Rate[/b]\n\n{win_rate:.2f}%")
            
            profit_factor = kpis['gross_profit'] / abs(kpis['gross_loss']) if kpis['gross_loss'] and kpis['gross_loss'] != 0 else float('inf')
            self.query_one("#kpi_profit_factor").update(f"[b]Profit Factor[/b]\n\n{profit_factor:.2f}")

            # --- Update Tables ---
            open_pos_table = self.query_one("#open_positions_table")
            open_pos_table.clear()
            for pos in open_positions:
                # Note: PnL for open positions is not live, it's the last updated value.
                open_pos_table.add_row(pos['symbol'], pos['side'], f"{pos['size']}", f"{pos['entry_price']}", f"{pos['pnl'] or 0:.2f}")

            recent_trades_table = self.query_one("#recent_trades_table")
            recent_trades_table.clear()
            for trade in recent_trades:
                recent_trades_table.add_row(trade['symbol'], f"{trade['closed_at']:%Y-%m-%d %H:%M}", f"{trade['pnl']:.2f}", trade['exit_reason'], f"{trade['holding_minutes']:.1f}")

        except Exception as e:
            self.query_one("#kpi_pnl").update(f"[b]Total PnL[/b]\n\n[red]ERROR: {e}[/red]")

if __name__ == "__main__":
    app = DashboardApp()
    app.run()