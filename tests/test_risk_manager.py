import unittest
from datetime import date, timedelta
from unittest.mock import patch

from agent.risk_manager import RiskManager, TradeAction


def _make_trade(
    ticker="AAPL",
    action="buy",
    quantity=10,
    reasoning="test",
    confidence=0.8,
    urgency="medium",
):
    return TradeAction(
        ticker=ticker,
        action=action,
        quantity=quantity,
        reasoning=reasoning,
        confidence=confidence,
        urgency=urgency,
    )


def _base_portfolio(
    cash=50000.0,
    total_value=100000.0,
    positions=None,
    daily_pnl=0.0,
):
    return {
        "cash": cash,
        "total_value": total_value,
        "positions": positions if positions is not None else [],
        "daily_pnl": daily_pnl,
    }


class TestTradeActionDataclass(unittest.TestCase):
    def test_fields_exist(self):
        trade = TradeAction(
            ticker="TSLA",
            action="buy",
            quantity=5,
            reasoning="momentum",
            confidence=0.75,
            urgency="high",
        )
        self.assertEqual(trade.ticker, "TSLA")
        self.assertEqual(trade.action, "buy")
        self.assertEqual(trade.quantity, 5)
        self.assertEqual(trade.reasoning, "momentum")
        self.assertAlmostEqual(trade.confidence, 0.75)
        self.assertEqual(trade.urgency, "high")


class TestRiskManagerValidateTrades(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()

    # Rule 1: Minimum confidence (buys only)
    def test_low_confidence_buy_rejected(self):
        trade = _make_trade(action="buy", confidence=0.4)
        result = self.rm.validate_trades([trade], _base_portfolio())
        self.assertEqual(result, [])

    def test_high_confidence_buy_approved(self):
        trade = _make_trade(action="buy", confidence=0.8)
        portfolio = _base_portfolio(
            cash=90000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 10, "current_price": 150.0, "avg_cost": 140.0}
            ],
        )
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ticker, "AAPL")

    def test_sell_always_passes_confidence(self):
        trade = _make_trade(action="sell", confidence=0.1)
        result = self.rm.validate_trades([trade], _base_portfolio())
        self.assertEqual(len(result), 1)

    # Rule 2: Max trades per day (buys only)
    def test_max_trades_per_day_blocks_buys(self):
        from config.settings import MAX_TRADES_PER_DAY
        self.rm._trades_today = MAX_TRADES_PER_DAY
        trade = _make_trade(action="buy", confidence=0.9)
        result = self.rm.validate_trades([trade], _base_portfolio())
        self.assertEqual(result, [])

    def test_max_trades_per_day_allows_sells(self):
        self.rm._trades_today = 10
        trade = _make_trade(action="sell", confidence=0.9)
        result = self.rm.validate_trades([trade], _base_portfolio())
        self.assertEqual(len(result), 1)

    # Rule 3: Daily loss circuit breaker
    def test_daily_loss_circuit_breaker(self):
        portfolio = _base_portfolio(total_value=100000.0, daily_pnl=-3500.0)
        buy_trade = _make_trade(action="buy", confidence=0.9)
        sell_trade = _make_trade(action="sell", confidence=0.1)
        result = self.rm.validate_trades([buy_trade, sell_trade], portfolio)
        # Circuit breaker blocks new buys but allows sells (loss-cutting)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].action, "sell")

    # Rule 4: VIX circuit breaker (blocks buys only)
    def test_vix_circuit_breaker_blocks_buys(self):
        macro = {"vix": 40.0}
        buy_trade = _make_trade(ticker="AAPL", action="buy", confidence=0.9)
        sell_trade = _make_trade(ticker="MSFT", action="sell", confidence=0.1)
        result = self.rm.validate_trades(
            [buy_trade, sell_trade], _base_portfolio(), macro_snapshot=macro
        )
        tickers = [t.ticker for t in result]
        self.assertNotIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)

    def test_vix_below_threshold_allows_buys(self):
        macro = {"vix": 20.0}
        trade = _make_trade(action="buy", confidence=0.9)
        portfolio = _base_portfolio(
            cash=90000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 10, "current_price": 150.0, "avg_cost": 140.0}
            ],
        )
        result = self.rm.validate_trades([trade], portfolio, macro_snapshot=macro)
        self.assertEqual(len(result), 1)

    # Rule 5: Cash reserve check (buys only)
    def test_cash_reserve_check(self):
        # cash=4000, total=100000; reserve=5000 (5%); available=max(0,4000-5000)=0 → rejected
        portfolio = _base_portfolio(
            cash=4000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 150.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(result, [])

    def test_cash_reserve_trims_buy_quantity(self):
        # cash=6000, total=100000; reserve=5000 (5%); available=1000; price=150 → max_qty=6
        # trade requests 10 → trimmed to 6
        portfolio = _base_portfolio(
            cash=6000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 150.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].quantity, 6)

    def test_cash_reserve_sufficient_allows_buy(self):
        # 1 share at $10 = $10 cost; cash=20000, total=100000
        # reserve required = 10000; cash after buy = 19990 > 10000 → approved
        portfolio = _base_portfolio(
            cash=20000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 10.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=1, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(len(result), 1)

    # Rule 6: Max position size (buys only) — trims quantity instead of rejecting
    def test_max_position_size(self):
        # total_value=100000; buy 350 shares at $110 = $38500 (38.5% > 30%)
        # max_qty_by_position = floor(100000 * 0.30 / 110) = 272
        # cash=90000 → available=85000 → max_qty_by_cash=772 (not limiting)
        # final_qty = min(350, 772, 272) = 272 → trimmed
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 110.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=350, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].quantity, 272)  # floor(100000 * 0.30 / 110)

    def test_max_position_size_fully_at_cap_rejected(self):
        # Existing position already at 30% cap → room=0 → rejected
        portfolio = _base_portfolio(
            cash=70000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 273, "current_price": 110.0, "avg_cost": 110.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(result, [])

    def test_max_position_size_within_limit_approved(self):
        # total_value=100000; buy 10 shares at $100 = $1000 → 1% of portfolio → approved
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 100.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        self.assertEqual(len(result), 1)

    # Rule 7: Concentration limit
    def test_concentration_limit(self):
        positions = [
            {"ticker": f"TICK{i}", "quantity": 1, "current_price": 100.0, "avg_cost": 100.0}
            for i in range(10)
        ]
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=positions,
        )
        new_trade = _make_trade(ticker="NEWCO", action="buy", quantity=1, confidence=0.9)
        result = self.rm.validate_trades([new_trade], portfolio)
        self.assertEqual(result, [])

    def test_concentration_limit_existing_ticker_allowed(self):
        positions = [
            {"ticker": f"TICK{i}", "quantity": 1, "current_price": 100.0, "avg_cost": 100.0}
            for i in range(10)
        ]
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=positions,
        )
        # Adding to an existing position should not trigger concentration limit
        existing_trade = _make_trade(ticker="TICK0", action="buy", quantity=1, confidence=0.9)
        result = self.rm.validate_trades([existing_trade], portfolio)
        self.assertEqual(len(result), 1)

    # Rule 8: Earnings blackout
    def test_earnings_blackout(self):
        today = date.today()
        earnings_date = today + timedelta(days=1)
        fundamentals = {
            "AAPL": {"next_earnings_date": earnings_date.isoformat()}
        }
        trade = _make_trade(ticker="AAPL", action="buy", confidence=0.9)
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 100.0, "avg_cost": 0.0}
            ],
        )
        result = self.rm.validate_trades([trade], portfolio, fundamentals=fundamentals)
        self.assertEqual(result, [])

    def test_earnings_blackout_far_future_allowed(self):
        today = date.today()
        far_future = today + timedelta(days=30)
        fundamentals = {
            "AAPL": {"next_earnings_date": far_future.isoformat()}
        }
        trade = _make_trade(ticker="AAPL", action="buy", confidence=0.9)
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 100.0, "avg_cost": 0.0}
            ],
        )
        result = self.rm.validate_trades([trade], portfolio, fundamentals=fundamentals)
        self.assertEqual(len(result), 1)

    def test_hold_actions_are_filtered_out(self):
        hold_trade = _make_trade(action="hold", confidence=0.9)
        result = self.rm.validate_trades([hold_trade], _base_portfolio())
        self.assertEqual(result, [])

    def test_daily_counter_resets_on_new_day(self):
        from datetime import date as _date, timedelta
        self.rm._trades_today = 5
        yesterday = _date.today() - timedelta(days=1)
        self.rm._trade_date = yesterday
        trade = _make_trade(action="sell", confidence=0.9)
        result = self.rm.validate_trades([trade], _base_portfolio())
        self.assertEqual(len(result), 1)
        # Sells don't increment the buy counter (reset happened + sell approved)
        self.assertEqual(self.rm._trades_today, 0)


if __name__ == "__main__":
    unittest.main()
