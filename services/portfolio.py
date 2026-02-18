import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.settings import INITIAL_CAPITAL
from db.database import (
    get_positions,
    update_position,
    insert_portfolio_snapshot,
    get_latest_portfolio_snapshot,
    get_day_open_snapshot,
    get_trades,
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    cash: float
    buying_power: float       # actual purchasing capacity (from Alpaca)
    positions: list[dict]
    total_value: float
    positions_value: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float


class PortfolioService:
    def __init__(self):
        self.realized_pnl = 0.0
        self.buying_power = 0.0
        self._load_state()

    def _load_state(self):
        """Load last known state from DB, or initialize with INITIAL_CAPITAL."""
        snapshot = get_latest_portfolio_snapshot()
        if snapshot:
            self.cash = snapshot["cash"]
            self.realized_pnl = snapshot.get("realized_pnl", 0.0)
        else:
            self.cash = INITIAL_CAPITAL

    def get_state(self, current_prices: dict[str, float]) -> PortfolioState:
        """Get current portfolio state with live prices."""
        positions = get_positions()
        enriched = []
        positions_value = 0.0
        unrealized_pnl = 0.0

        for pos in positions:
            price = current_prices.get(
                pos["ticker"],
                pos.get("current_price") or pos["avg_cost"],
            )
            market_value = pos["quantity"] * price
            pnl = (price - pos["avg_cost"]) * pos["quantity"]
            enriched.append({
                "ticker": pos["ticker"],
                "quantity": pos["quantity"],
                "avg_cost": pos["avg_cost"],
                "current_price": price,
                "market_value": market_value,
                "unrealized_pnl": pnl,
            })
            positions_value += market_value
            unrealized_pnl += pnl

        total_value = self.cash + positions_value

        # Daily P&L: compare against today's opening value (first snapshot of the day)
        from datetime import date as _date
        today_str = _date.today().isoformat()
        day_open = get_day_open_snapshot(today_str)
        if day_open:
            starting_value = day_open["total_value"]
        else:
            starting_value = INITIAL_CAPITAL
        daily_pnl = total_value - starting_value

        return PortfolioState(
            cash=self.cash,
            buying_power=self.buying_power,
            positions=enriched,
            total_value=total_value,
            positions_value=positions_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
        )

    def record_buy(self, ticker: str, quantity: int, price: float):
        """Record a buy trade -- update cash and position."""
        cost = quantity * price
        if cost > self.cash:
            logger.warning(
                "Insufficient cash to buy %d shares of %s at %.2f "
                "(cost=%.2f, available=%.2f); trade rejected",
                quantity,
                ticker,
                price,
                cost,
                self.cash,
            )
            return

        positions = get_positions()
        existing = next((p for p in positions if p["ticker"] == ticker), None)

        if existing:
            new_quantity = existing["quantity"] + quantity
            new_avg_cost = (
                (existing["avg_cost"] * existing["quantity"]) + (price * quantity)
            ) / new_quantity
        else:
            new_quantity = quantity
            new_avg_cost = price

        self.cash -= cost
        update_position(ticker, new_quantity, new_avg_cost)

        logger.info(
            "BUY recorded: %d shares of %s at %.2f (total=%.2f, cash_remaining=%.2f)",
            quantity,
            ticker,
            price,
            cost,
            self.cash,
        )

    def record_sell(self, ticker: str, quantity: int, price: float):
        """Record a sell trade -- update cash, position, and realized P&L."""
        positions = get_positions()
        existing = next((p for p in positions if p["ticker"] == ticker), None)

        if not existing or existing["quantity"] <= 0:
            logger.warning(
                "Cannot sell %s: no position held; trade rejected",
                ticker,
            )
            return

        if quantity > existing["quantity"]:
            logger.warning(
                "Sell quantity %d exceeds held quantity %.2f for %s; capping at held quantity",
                quantity,
                existing["quantity"],
                ticker,
            )
            quantity = int(existing["quantity"])

        proceeds = quantity * price
        pnl = (price - existing["avg_cost"]) * quantity

        self.cash += proceeds
        self.realized_pnl += pnl

        new_quantity = existing["quantity"] - quantity
        update_position(ticker, new_quantity, existing["avg_cost"])

        logger.info(
            "SELL recorded: %d shares of %s at %.2f "
            "(proceeds=%.2f, realized_pnl=%.2f, cash=%.2f)",
            quantity,
            ticker,
            price,
            proceeds,
            pnl,
            self.cash,
        )

    def sync_from_alpaca(self, account: dict, alpaca_positions: list[dict]) -> None:
        """Sync portfolio state from Alpaca (source of truth).

        Called after trade execution and before each snapshot so the local DB
        always reflects the real Alpaca paper account, regardless of whether
        filled_avg_price was available at order submission time.
        """
        self.cash = float(account.get("cash", self.cash))
        self.buying_power = float(account.get("buying_power", 0.0))

        # Recompute realized P&L: equity - initial_capital - total_unrealized
        equity = float(account.get("equity", self.cash))
        total_unrealized = sum(float(p.get("unrealized_pl", 0)) for p in alpaca_positions)
        self.realized_pnl = equity - INITIAL_CAPITAL - total_unrealized

        # Sync positions table: upsert from Alpaca, remove positions no longer held
        existing_tickers = {p["ticker"] for p in get_positions()}
        alpaca_tickers = set()
        for pos in alpaca_positions:
            ticker = pos["ticker"]
            alpaca_tickers.add(ticker)
            update_position(
                ticker,
                pos["quantity"],
                pos["avg_entry_price"],
                pos.get("current_price"),
            )

        for ticker in existing_tickers - alpaca_tickers:
            update_position(ticker, 0, 0)  # removes the row

        logger.info(
            "Synced from Alpaca: cash=%.2f, equity=%.2f, positions=%d, realized_pnl=%.2f",
            self.cash, equity, len(alpaca_positions), self.realized_pnl,
        )

    def take_snapshot(self, current_prices: dict[str, float]):
        """Save current portfolio state to DB."""
        state = self.get_state(current_prices)
        insert_portfolio_snapshot({
            "total_value": state.total_value,
            "cash": state.cash,
            "positions_value": state.positions_value,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": state.unrealized_pnl,
            "daily_pnl": state.daily_pnl,
        })
        logger.info(
            "Portfolio snapshot saved: total_value=%.2f, cash=%.2f, "
            "positions_value=%.2f, unrealized_pnl=%.2f",
            state.total_value,
            state.cash,
            state.positions_value,
            state.unrealized_pnl,
        )
