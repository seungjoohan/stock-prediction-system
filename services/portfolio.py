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
        self._alpaca_unrealized = 0.0
        self.equity = INITIAL_CAPITAL
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
        """Record a buy trade -- update cash and position.

        Handles three cases:
        1. New long position (no existing position)
        2. Adding to an existing long position
        3. Buy-to-cover (existing qty < 0): covers do not consume cash;
           they reduce/eliminate a short liability.
        """
        positions = get_positions()
        existing = next((p for p in positions if p["ticker"] == ticker), None)

        existing_qty = existing["quantity"] if existing else 0.0
        is_cover = existing_qty < 0

        cost = quantity * price

        # Cash check only applies to genuine new buys, not buy-to-cover.
        # Covering a short returns collateral rather than spending cash.
        if not is_cover and cost > self.cash:
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

        if existing:
            new_quantity = existing["quantity"] + quantity
            if new_quantity == 0:
                # Full cover: position is flat — avg_cost is irrelevant
                new_avg_cost = 0.0
            elif new_quantity < 0:
                # Partial cover: still short — preserve original entry price
                new_avg_cost = existing["avg_cost"]
            elif is_cover:
                # Over-cover: flipped from short to long — cost basis is the cover price only
                new_avg_cost = price
            else:
                # Adding to an existing long position
                new_avg_cost = (
                    (existing["avg_cost"] * existing["quantity"]) + (price * quantity)
                ) / new_quantity
        else:
            new_quantity = quantity
            new_avg_cost = price

        # For covers, cash does not change (Alpaca handles collateral return on settlement).
        # For genuine buys, deduct the cost.
        if not is_cover:
            self.cash -= cost

        update_position(ticker, new_quantity, new_avg_cost)

        logger.info(
            "BUY recorded: %d shares of %s at %.2f (cost=%.2f, is_cover=%s, cash_remaining=%.2f)",
            quantity,
            ticker,
            price,
            cost,
            is_cover,
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

        avg_cost_at_time = existing["avg_cost"]
        proceeds = quantity * price
        pnl = (price - avg_cost_at_time) * quantity

        self.cash += proceeds
        self.realized_pnl += pnl

        new_quantity = existing["quantity"] - quantity
        update_position(ticker, new_quantity, avg_cost_at_time)

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
        self._alpaca_unrealized = total_unrealized
        self.equity = equity

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
        """Save current portfolio state to DB.

        Uses Alpaca equity as total_value (source of truth) so that short
        positions are correctly reflected.  self.equity and self._alpaca_unrealized
        are populated by sync_from_alpaca() before this is called.
        """
        # Alpaca equity = cash + net market value of all positions (longs + shorts).
        # Using it directly avoids the long-only bias from the local positions table.
        total_value = self.equity if self.equity else (self.cash + self._alpaca_unrealized)
        positions_value = total_value - self.cash

        # Daily P&L: compare against today's opening equity snapshot.
        from datetime import date as _date
        today_str = _date.today().isoformat()
        day_open = get_day_open_snapshot(today_str)
        starting_value = day_open["total_value"] if day_open else INITIAL_CAPITAL
        daily_pnl = total_value - starting_value

        insert_portfolio_snapshot({
            "total_value": total_value,
            "cash": self.cash,
            "positions_value": positions_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self._alpaca_unrealized,
            "daily_pnl": daily_pnl,
        })
        logger.info(
            "Portfolio snapshot saved: total_value=%.2f, cash=%.2f, "
            "positions_value=%.2f, unrealized_pnl=%.2f",
            total_value,
            self.cash,
            positions_value,
            self._alpaca_unrealized,
        )
