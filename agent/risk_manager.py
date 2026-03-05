import dataclasses
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
    ALLOW_SHORT_SELLING,
    MAX_SHORT_POSITIONS,
    MAX_SHORT_EXPOSURE_PCT,
    MIN_SHORT_CONFIDENCE,
    MAX_SHORT_POSITION_PCT,
    MAX_DRAWDOWN_PCT,
    MAX_SECTOR_EXPOSURE_PCT,
    SECTOR_MAP,
)
from db.database import insert_agent_log, get_trades, get_max_equity_since

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
        self._trade_date = date.today()
        self._trades_today = self._count_buys_today()

    def _count_buys_today(self) -> int:
        """Count today's buy trades from the database."""
        today_str = self._trade_date.isoformat()
        trades = get_trades(limit=200)
        return sum(
            1 for t in trades
            if t.get("action") == "buy" and t.get("timestamp", "").startswith(today_str)
        )

    def validate_trades(
        self,
        trades: list[TradeAction],
        portfolio_state: dict,
        macro_snapshot: dict | None = None,
        fundamentals: dict | None = None,
        current_prices: dict | None = None,
        alpaca_positions: list[dict] | None = None,
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
        # Merge: forced stop-loss actions always take priority over LLM proposals.
        # Remove any LLM proposal for the same ticker so the forced action (full
        # position exit, confidence=1.0, urgency="immediate") is not silently
        # replaced by a smaller or lower-confidence LLM trade on the same ticker.
        forced_tickers = {f.ticker for f in forced}
        if forced_tickers:
            dropped = [t for t in candidates if t.ticker in forced_tickers]
            for t in dropped:
                logger.warning(
                    "LLM proposal %s %s qty=%d overridden by forced stop-loss action",
                    t.action, t.ticker, t.quantity,
                )
            candidates = [t for t in candidates if t.ticker not in forced_tickers]
        for f in forced:
            candidates.insert(0, f)

        approved: list[TradeAction] = []
        # Track cash committed by buys approved earlier in this same cycle so
        # subsequent approvals see the reduced available balance, preventing
        # multiple trades from collectively spending more than the actual cash.
        committed_cash = 0.0

        daily_loss_exceeded = self._check_daily_loss(portfolio_state)

        # Rolling drawdown circuit breaker: check if current equity has dropped
        # more than MAX_DRAWDOWN_PCT below the 20-day high.
        drawdown_halt = False
        current_equity = portfolio_state.get("total_value", 0)
        peak_equity = get_max_equity_since(days=20)
        if peak_equity and peak_equity > 0 and current_equity > 0:
            drawdown_pct = (peak_equity - current_equity) / peak_equity
            if drawdown_pct >= MAX_DRAWDOWN_PCT:
                drawdown_halt = True
                logger.warning(
                    "DRAWDOWN HALT: equity %.2f is %.1f%% below 20-day peak %.2f (threshold %.0f%%)",
                    current_equity, drawdown_pct * 100, peak_equity, MAX_DRAWDOWN_PCT * 100,
                )

        for trade in candidates:
            is_sell = trade.action == "sell"

            # Detect buy-to-cover: a buy against a ticker currently held short.
            # Covers must bypass buy-side restrictions (cash, daily limits, VIX, concentration)
            # because they reduce risk rather than adding new exposure.
            short_qty = 0.0
            if not is_sell:
                for pos in portfolio_state.get("positions", []):
                    if pos.get("ticker") == trade.ticker and pos.get("quantity", 0) < 0:
                        short_qty = pos["quantity"]
                        break
            is_cover = not is_sell and short_qty < 0

            # Rule 1: Minimum confidence (new buys only — not covers or sells)
            if not is_sell and not is_cover and trade.confidence < MIN_TRADE_CONFIDENCE:
                logger.info(
                    "REJECTED %s %s: confidence %.2f below minimum %.2f",
                    trade.action,
                    trade.ticker,
                    trade.confidence,
                    MIN_TRADE_CONFIDENCE,
                )
                continue

            # Rule 2: Max trades per day (new buys only — covers and sells are always allowed)
            if not is_sell and not is_cover and self._trades_today >= MAX_TRADES_PER_DAY:
                logger.info(
                    "REJECTED buy %s: daily trade limit %d reached",
                    trade.ticker,
                    MAX_TRADES_PER_DAY,
                )
                continue

            # Rule 3: Max daily loss circuit breaker (blocks new buys only — not covers)
            if not is_sell and not is_cover and daily_loss_exceeded:
                logger.info(
                    "REJECTED %s %s: daily loss circuit breaker triggered (loss >= %.1f%%)",
                    trade.action,
                    trade.ticker,
                    MAX_DAILY_LOSS_PCT * 100,
                )
                continue

            # Rule 3b: Rolling drawdown circuit breaker (blocks new entries only)
            if not is_sell and not is_cover and drawdown_halt:
                logger.info(
                    "REJECTED %s %s: rolling drawdown circuit breaker active (>= %.0f%% below 20-day peak)",
                    trade.action, trade.ticker, MAX_DRAWDOWN_PCT * 100,
                )
                continue

            # Rule 4: VIX circuit breaker (blocks new buys only — not covers)
            if not is_sell and not is_cover and macro_snapshot is not None:
                vix = macro_snapshot.get("vix")
                if vix is not None and vix > VIX_CIRCUIT_BREAKER:
                    logger.info(
                        "REJECTED buy %s: VIX %.1f exceeds circuit breaker %.1f",
                        trade.ticker,
                        vix,
                        VIX_CIRCUIT_BREAKER,
                    )
                    continue

            # Rule 5+6: Trim buy quantity to fit within cash reserve AND position size limits
            # Covers bypass this entirely — they return collateral rather than spending cash.
            if not is_sell and not is_cover:
                total_value = portfolio_state.get("total_value", 0)
                # Use actual cash, not buying_power — buying_power can include Alpaca margin
                # (2× cash on margin accounts), which would let the risk manager approve trades
                # that record_buy() then silently rejects because cost > self.cash.
                cash = max(0.0, portfolio_state.get("cash", 0))
                price = (
                    (current_prices or {}).get(trade.ticker)
                    or self._get_current_price(trade.ticker, portfolio_state)
                )
                if price and price > 0:
                    # Max quantity by cash reserve: keep CASH_RESERVE_PCT of total_value in cash.
                    # Subtract committed_cash so earlier approvals in this cycle reduce the
                    # budget for later trades, preventing collective overspend.
                    required_reserve = total_value * CASH_RESERVE_PCT
                    available_to_spend = max(0.0, cash - required_reserve - committed_cash)
                    max_qty_by_cash = int(available_to_spend / price)

                    # Max quantity by position size: single ticker <= MAX_POSITION_PCT of total_value
                    existing_value = 0.0
                    for pos in portfolio_state.get("positions", []):
                        if pos.get("ticker") == trade.ticker:
                            existing_value = abs(pos.get("quantity", 0)) * (
                                pos.get("current_price") or pos.get("avg_cost") or price
                            )
                            break
                    max_position_value = total_value * MAX_POSITION_PCT
                    room_remaining = max(0.0, max_position_value - existing_value)
                    max_qty_by_position = int(room_remaining / price)

                    # Final quantity: minimum of requested, cash limit, and position limit
                    final_qty = min(trade.quantity, max_qty_by_cash, max_qty_by_position)

                    if final_qty <= 0:
                        if max_qty_by_position <= 0:
                            logger.info(
                                "REJECTED buy %s: already at max position size %.1f%% of portfolio",
                                trade.ticker, MAX_POSITION_PCT * 100,
                            )
                        else:
                            logger.info(
                                "REJECTED buy %s: insufficient cash after %.1f%% reserve "
                                "(cash=%.2f, reserve=%.2f)",
                                trade.ticker, CASH_RESERVE_PCT * 100, cash, required_reserve,
                            )
                        continue

                    if final_qty < trade.quantity:
                        logger.info(
                            "TRIMMED buy %s: quantity %d → %d (cash_limit=%d, position_limit=%d)",
                            trade.ticker, trade.quantity, final_qty, max_qty_by_cash, max_qty_by_position,
                        )
                        trade = TradeAction(
                            ticker=trade.ticker,
                            action=trade.action,
                            quantity=final_qty,
                            reasoning=trade.reasoning,
                            confidence=trade.confidence,
                            urgency=trade.urgency,
                        )
                # If price is unknown, skip size/cash enforcement and proceed to Rule 7+8

            # Rule 7: Concentration limit — new buys only (covers reduce exposure, not add it)
            if not is_sell and not is_cover:
                positions = portfolio_state.get("positions", [])
                # Include both longs (qty > 0) and shorts (qty < 0) in occupied-slot count
                held_tickers = {p["ticker"] for p in positions if p.get("quantity", 0) != 0}
                ticker_is_new = trade.ticker not in held_tickers
                if ticker_is_new and len(held_tickers) >= MAX_POSITIONS:
                    logger.info(
                        "REJECTED buy %s: portfolio already at max %d positions",
                        trade.ticker,
                        MAX_POSITIONS,
                    )
                    continue

            # Rule 8: Earnings blackout (no new buys within blackout window — not covers)
            if not is_sell and not is_cover and self._check_earnings_blackout(trade.ticker, fundamentals):
                logger.info(
                    "REJECTED buy %s: within %d-day earnings blackout period",
                    trade.ticker,
                    EARNINGS_BLACKOUT_DAYS,
                )
                continue

            # Rule 9: Sector exposure limit (new buys only)
            if not is_sell and not is_cover:
                trade_sector = SECTOR_MAP.get(trade.ticker)
                if trade_sector and trade_sector != "ETF":
                    total_value = portfolio_state.get("total_value", 0)
                    if total_value > 0:
                        sector_value = 0.0
                        for pos in portfolio_state.get("positions", []):
                            pticker = pos.get("ticker")
                            pqty = pos.get("quantity", 0)
                            if pqty > 0 and SECTOR_MAP.get(pticker) == trade_sector:
                                pprice = pos.get("current_price") or pos.get("avg_cost", 0)
                                sector_value += pqty * pprice
                        price = (
                            (current_prices or {}).get(trade.ticker)
                            or self._get_current_price(trade.ticker, portfolio_state)
                        )
                        if price and price > 0:
                            new_sector_value = sector_value + trade.quantity * price
                            if new_sector_value / total_value > MAX_SECTOR_EXPOSURE_PCT:
                                logger.info(
                                    "REJECTED buy %s: sector '%s' exposure would reach %.1f%% (limit %.0f%%)",
                                    trade.ticker, trade_sector,
                                    (new_sector_value / total_value) * 100,
                                    MAX_SECTOR_EXPOSURE_PCT * 100,
                                )
                                continue

            # Detect short entry: sell quantity > held position
            if trade.action == "sell" and alpaca_positions is not None:
                held = next(
                    (float(p["quantity"]) for p in alpaca_positions if p["ticker"] == trade.ticker),
                    0.0,
                )
                if trade.quantity > held:
                    # Block new short entries during drawdown halt
                    if drawdown_halt:
                        logger.info(
                            "REJECTED short %s: rolling drawdown circuit breaker active",
                            trade.ticker,
                        )
                        if held > 0:
                            trade = dataclasses.replace(trade, quantity=int(held))
                        else:
                            continue

                if trade.quantity > held:
                    short_ok, reason = self._validate_short_entry(
                        trade, held, alpaca_positions,
                        equity=portfolio_state.get("total_value", 0),
                        current_prices=current_prices,
                    )
                    if not short_ok:
                        logger.warning("Short entry rejected for %s: %s", trade.ticker, reason)
                        insert_agent_log("short_rejected", {"ticker": trade.ticker, "reason": reason}, trade.ticker)
                        if held > 0:
                            trade = dataclasses.replace(trade, quantity=int(held))
                        else:
                            continue

            approved.append(trade)
            if not is_sell and not is_cover:
                self._trades_today += 1
                # Deduct the cost from the intra-cycle cash budget so subsequent
                # buys in the same decision cycle see reduced available cash.
                buy_price = (
                    (current_prices or {}).get(trade.ticker)
                    or self._get_current_price(trade.ticker, portfolio_state)
                )
                if buy_price and buy_price > 0:
                    committed_cash += trade.quantity * buy_price

        return approved

    def _validate_short_entry(
        self,
        trade,
        held_qty: float,
        alpaca_positions: list[dict],
        equity: float,
        current_prices: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Called when a sell would open or expand a short position.
        """
        if not ALLOW_SHORT_SELLING:
            return False, f"Short selling disabled — sell capped at held qty {held_qty:.0f}"

        current_shorts = [p for p in alpaca_positions if float(p.get("quantity", 0)) < 0]
        ticker_already_short = any(p["ticker"] == trade.ticker for p in current_shorts)
        if not ticker_already_short and len(current_shorts) >= MAX_SHORT_POSITIONS:
            return False, f"Max short positions ({MAX_SHORT_POSITIONS}) reached"

        short_exposure = sum(abs(float(p["quantity"]) * float(p.get("current_price", p.get("avg_entry_price", 0))))
                             for p in current_shorts if p["ticker"] != trade.ticker)
        # Use live market price (not trade.price — TradeAction has no price field)
        entry_price = (current_prices or {}).get(trade.ticker, 0.0)
        new_short_value = trade.quantity * entry_price
        if equity > 0 and (short_exposure + new_short_value) / equity > MAX_SHORT_EXPOSURE_PCT:
            return False, f"Short exposure would exceed {MAX_SHORT_EXPOSURE_PCT:.0%} of equity"

        if trade.confidence < MIN_SHORT_CONFIDENCE:
            return False, f"Confidence {trade.confidence:.2f} below short minimum {MIN_SHORT_CONFIDENCE:.2f}"

        # Per-position short size limit
        if equity > 0 and entry_price > 0:
            max_short_value = equity * MAX_SHORT_POSITION_PCT
            if new_short_value > max_short_value:
                trimmed_qty = int(max_short_value / entry_price)
                if trimmed_qty <= 0:
                    return False, (
                        f"Short position value would exceed {MAX_SHORT_POSITION_PCT:.0%} of equity "
                        f"and cannot be trimmed"
                    )
                logger.info(
                    "TRIMMED short %s: quantity %d → %d (per-position limit %.0f%% of equity)",
                    trade.ticker, trade.quantity, trimmed_qty, MAX_SHORT_POSITION_PCT * 100,
                )
                trade.quantity = trimmed_qty

        return True, "short approved"

    def _check_daily_loss(self, portfolio_state: dict) -> bool:
        """Return True if daily loss exceeds threshold."""
        total_value = portfolio_state.get("total_value", 0)
        daily_pnl = portfolio_state.get("daily_pnl", 0)
        if total_value <= 0:
            return False
        loss_pct = -daily_pnl / total_value
        return loss_pct >= MAX_DAILY_LOSS_PCT

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
        """Return forced TradeActions for any position (long or short) that hits stop-loss."""
        forced = []
        for pos in portfolio_state.get("positions", []):
            ticker = pos.get("ticker")
            qty = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0)
            if not ticker or qty == 0 or avg_cost <= 0:
                continue

            price = current_prices.get(ticker) or pos.get("current_price", 0)
            if not price:
                continue

            if qty > 0:
                # Long: loss when price drops below avg_cost
                pnl_pct = (price - avg_cost) / avg_cost
                if pnl_pct <= -STOP_LOSS_PCT:
                    logger.warning(
                        "STOP-LOSS triggered for %s (LONG): down %.1f%% (price=%.2f, avg_cost=%.2f)",
                        ticker, pnl_pct * 100, price, avg_cost,
                    )
                    forced.append(TradeAction(
                        ticker=ticker,
                        action="sell",
                        quantity=int(qty),
                        reasoning=f"Stop-loss: long position down {pnl_pct:.1%} vs threshold -{STOP_LOSS_PCT:.0%}",
                        confidence=1.0,
                        urgency="immediate",
                    ))
            else:
                # Short: loss when price rises above avg_cost (short entry price)
                pnl_pct = (avg_cost - price) / avg_cost
                if pnl_pct <= -STOP_LOSS_PCT:
                    logger.warning(
                        "STOP-LOSS triggered for %s (SHORT): down %.1f%% (price=%.2f, avg_entry=%.2f)",
                        ticker, pnl_pct * 100, price, avg_cost,
                    )
                    forced.append(TradeAction(
                        ticker=ticker,
                        action="buy",
                        quantity=int(abs(qty)),
                        reasoning=f"Stop-loss: short position down {pnl_pct:.1%} vs threshold -{STOP_LOSS_PCT:.0%}",
                        confidence=1.0,
                        urgency="immediate",
                    ))
        return forced
