import logging
from dataclasses import dataclass
from datetime import date, timedelta

from config.settings import (
    MAX_POSITION_PCT,
    MAX_DAILY_LOSS_PCT,
    MIN_TRADE_CONFIDENCE,
    MAX_TRADES_PER_DAY,
    CASH_RESERVE_PCT,
    MAX_POSITIONS,
    VIX_CIRCUIT_BREAKER,
    EARNINGS_BLACKOUT_DAYS,
    STOP_LOSS_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeAction:
    ticker: str
    action: str         # "buy" | "sell" | "hold"
    quantity: int
    reasoning: str
    confidence: float
    urgency: str        # "low" | "medium" | "high"


class RiskManager:
    def __init__(self):
        self._trades_today = 0
        self._trade_date = date.today()

    def validate_trades(
        self,
        trades: list[TradeAction],
        portfolio_state: dict,
        macro_snapshot: dict | None = None,
        fundamentals: dict | None = None,
        current_prices: dict | None = None,
    ) -> list[TradeAction]:
        """Apply all risk rules. Returns filtered list of approved trades."""
        today = date.today()
        if today != self._trade_date:
            self._trades_today = 0
            self._trade_date = today

        candidates = [t for t in trades if t.action != "hold"]

        # Inject forced sells for stop-loss / take-profit before LLM proposals
        forced = self._check_stop_loss_take_profit(
            portfolio_state, current_prices or {}
        )
        # Merge: forced sells take priority; don't duplicate tickers already proposed
        proposed_tickers = {t.ticker for t in candidates if t.action == "sell"}
        for sell in forced:
            if sell.ticker not in proposed_tickers:
                candidates.insert(0, sell)

        approved: list[TradeAction] = []

        daily_loss_exceeded = self._check_daily_loss(portfolio_state)

        for trade in candidates:
            is_sell = trade.action == "sell"

            # Rule 1: Minimum confidence (buys only)
            if not is_sell and trade.confidence < MIN_TRADE_CONFIDENCE:
                logger.info(
                    "REJECTED %s %s: confidence %.2f below minimum %.2f",
                    trade.action,
                    trade.ticker,
                    trade.confidence,
                    MIN_TRADE_CONFIDENCE,
                )
                continue

            # Rule 2: Max trades per day (buys only — sells are always allowed)
            if not is_sell and self._trades_today >= MAX_TRADES_PER_DAY:
                logger.info(
                    "REJECTED buy %s: daily trade limit %d reached",
                    trade.ticker,
                    MAX_TRADES_PER_DAY,
                )
                continue

            # Rule 3: Max daily loss circuit breaker (blocks new buys only)
            if not is_sell and daily_loss_exceeded:
                logger.info(
                    "REJECTED %s %s: daily loss circuit breaker triggered (loss >= %.1f%%)",
                    trade.action,
                    trade.ticker,
                    MAX_DAILY_LOSS_PCT * 100,
                )
                continue

            # Rule 4: VIX circuit breaker (blocks buys only)
            if not is_sell and macro_snapshot is not None:
                vix = macro_snapshot.get("vix")
                if vix is not None and vix > VIX_CIRCUIT_BREAKER:
                    logger.info(
                        "REJECTED buy %s: VIX %.1f exceeds circuit breaker %.1f",
                        trade.ticker,
                        vix,
                        VIX_CIRCUIT_BREAKER,
                    )
                    continue

            # Rule 5: Cash reserve check (buys only) — use buying_power (Alpaca margin-safe)
            if not is_sell:
                total_value = portfolio_state.get("total_value", 0)
                buying_power = portfolio_state.get(
                    "buying_power", portfolio_state.get("cash", 0)
                )
                required_reserve = total_value * CASH_RESERVE_PCT
                price = (
                    (current_prices or {}).get(trade.ticker)
                    or self._get_current_price(trade.ticker, portfolio_state)
                )
                position_cost = trade.quantity * price
                if buying_power - position_cost < required_reserve:
                    logger.info(
                        "REJECTED buy %s: insufficient buying power after %.1f%% reserve "
                        "(buying_power=%.2f, cost=%.2f, reserve=%.2f)",
                        trade.ticker,
                        CASH_RESERVE_PCT * 100,
                        buying_power,
                        position_cost,
                        required_reserve,
                    )
                    continue

            # Rule 6: Max position size check (buys only)
            if not is_sell and self._check_position_size(trade, portfolio_state, current_prices):
                logger.info(
                    "REJECTED buy %s: would exceed max position size %.1f%% of portfolio",
                    trade.ticker,
                    MAX_POSITION_PCT * 100,
                )
                continue

            # Rule 7: Concentration limit — max positions for new buys
            if not is_sell:
                positions = portfolio_state.get("positions", [])
                held_tickers = {p["ticker"] for p in positions if p.get("quantity", 0) > 0}
                ticker_is_new = trade.ticker not in held_tickers
                if ticker_is_new and len(held_tickers) >= MAX_POSITIONS:
                    logger.info(
                        "REJECTED buy %s: portfolio already at max %d positions",
                        trade.ticker,
                        MAX_POSITIONS,
                    )
                    continue

            # Rule 8: Earnings blackout (no new buys within blackout window)
            if not is_sell and self._check_earnings_blackout(trade.ticker, fundamentals):
                logger.info(
                    "REJECTED buy %s: within %d-day earnings blackout period",
                    trade.ticker,
                    EARNINGS_BLACKOUT_DAYS,
                )
                continue

            approved.append(trade)
            if not is_sell:
                self._trades_today += 1

        return approved

    def _check_daily_loss(self, portfolio_state: dict) -> bool:
        """Return True if daily loss exceeds threshold."""
        total_value = portfolio_state.get("total_value", 0)
        daily_pnl = portfolio_state.get("daily_pnl", 0)
        if total_value <= 0:
            return False
        loss_pct = -daily_pnl / total_value
        return loss_pct >= MAX_DAILY_LOSS_PCT

    def _check_position_size(
        self, trade: TradeAction, portfolio_state: dict, current_prices: dict | None = None
    ) -> bool:
        """Return True if trade would exceed max position size."""
        total_value = portfolio_state.get("total_value", 0)
        if total_value <= 0:
            return False

        current_price = (
            (current_prices or {}).get(trade.ticker)
            or self._get_current_price(trade.ticker, portfolio_state)
        )
        new_cost = trade.quantity * current_price

        existing_value = 0.0
        for pos in portfolio_state.get("positions", []):
            if pos.get("ticker") == trade.ticker:
                existing_value = pos.get("quantity", 0) * pos.get("current_price", 0)
                break

        projected_position_value = existing_value + new_cost
        return projected_position_value / total_value > MAX_POSITION_PCT

    def _check_earnings_blackout(self, ticker: str, fundamentals: dict | None) -> bool:
        """Return True if ticker is within earnings blackout period."""
        if fundamentals is None:
            return False

        ticker_data = fundamentals.get(ticker)
        if ticker_data is None:
            return False

        earnings_date_raw = ticker_data.get("next_earnings_date")
        if earnings_date_raw is None:
            return False

        if isinstance(earnings_date_raw, str):
            try:
                earnings_date = date.fromisoformat(earnings_date_raw)
            except ValueError:
                logger.warning(
                    "Could not parse earnings date for %s: %s",
                    ticker,
                    earnings_date_raw,
                )
                return False
        elif isinstance(earnings_date_raw, date):
            earnings_date = earnings_date_raw
        else:
            return False

        today = date.today()
        delta = (earnings_date - today).days
        return 0 <= delta <= EARNINGS_BLACKOUT_DAYS

    @staticmethod
    def _get_current_price(ticker: str, portfolio_state: dict) -> float:
        """Resolve current price for a ticker from portfolio positions."""
        for pos in portfolio_state.get("positions", []):
            if pos.get("ticker") == ticker:
                return float(pos.get("current_price", 0))
        return 0.0

    def _check_stop_loss_take_profit(
        self, portfolio_state: dict, current_prices: dict
    ) -> list[TradeAction]:
        """Return forced sell TradeActions for any position that hits stop-loss or take-profit."""
        forced = []
        for pos in portfolio_state.get("positions", []):
            ticker = pos.get("ticker")
            qty = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0)
            if not ticker or qty <= 0 or avg_cost <= 0:
                continue

            price = current_prices.get(ticker) or pos.get("current_price", 0)
            if not price:
                continue

            pnl_pct = (price - avg_cost) / avg_cost

            if pnl_pct <= -STOP_LOSS_PCT:
                logger.warning(
                    "STOP-LOSS triggered for %s: down %.1f%% (price=%.2f, avg_cost=%.2f)",
                    ticker, pnl_pct * 100, price, avg_cost,
                )
                forced.append(TradeAction(
                    ticker=ticker,
                    action="sell",
                    quantity=int(qty),
                    reasoning=f"Stop-loss: position down {pnl_pct:.1%} vs threshold -{STOP_LOSS_PCT:.0%}",
                    confidence=1.0,
                    urgency="immediate",
                ))
        return forced
