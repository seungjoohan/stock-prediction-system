import unittest
from datetime import date, timedelta
from unittest.mock import patch

_RM_MODULE = "agent.risk_manager"

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
        with patch("agent.risk_manager.get_trades", return_value=[]), \
             patch("agent.risk_manager.get_water_marks", return_value={}):
            self.rm = RiskManager()
        # Prevent drawdown halt from interfering with tests
        self._peak_patcher = patch("agent.risk_manager.get_max_equity_since", return_value=None)
        self._peak_patcher.start()
        self._upsert_wm_patcher = patch("agent.risk_manager.upsert_water_mark")
        self._upsert_wm_patcher.start()
        self._delete_wm_patcher = patch("agent.risk_manager.delete_water_mark")
        self._delete_wm_patcher.start()

    def tearDown(self):
        self._peak_patcher.stop()
        self._upsert_wm_patcher.stop()
        self._delete_wm_patcher.stop()

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
    @patch(f"{_RM_MODULE}.MAX_DAILY_LOSS_PCT", 0.03)
    def test_daily_loss_circuit_breaker(self):
        portfolio = _base_portfolio(
            total_value=100000.0,
            daily_pnl=-3500.0,
            positions=[{"ticker": "AAPL", "quantity": 50, "avg_cost": 150.0, "current_price": 150.0}],
        )
        buy_trade = _make_trade(action="buy", confidence=0.9)
        sell_trade = _make_trade(action="sell", confidence=0.1)
        result = self.rm.validate_trades([buy_trade, sell_trade], portfolio)
        # Circuit breaker blocks new buys but allows sells (loss-cutting)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].action, "sell")

    # Rule 4: VIX circuit breaker (blocks new buys and new short entries)
    def test_vix_circuit_breaker_blocks_buys(self):
        macro = {"vix": 40.0}
        buy_trade = _make_trade(ticker="AAPL", action="buy", confidence=0.9)
        # Sell MSFT where position is held long — normal sell should pass through
        sell_trade = _make_trade(ticker="MSFT", action="sell", quantity=10, confidence=0.1)
        portfolio = _base_portfolio(
            positions=[
                {"ticker": "MSFT", "quantity": 10, "current_price": 300.0, "avg_cost": 280.0},
            ],
        )
        result = self.rm.validate_trades(
            [buy_trade, sell_trade], portfolio, macro_snapshot=macro
        )
        tickers = [t.ticker for t in result]
        self.assertNotIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)

    def test_vix_tier3_blocks_short_entries(self):
        """VIX >= 30 (Tier 3) should block new short entries."""
        macro = {"vix": 32.0}
        # Sell 10 shares of AAPL with no long position held — this is a short entry
        short_trade = _make_trade(ticker="AAPL", action="sell", quantity=10, confidence=0.9)
        result = self.rm.validate_trades(
            [short_trade], _base_portfolio(), macro_snapshot=macro
        )
        self.assertEqual(len(result), 0)

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

    def test_multi_buy_committed_cash_prevents_overspend(self):
        # cash=7000, total=100000; reserve=5000 (5%); available=2000
        # Two buys each requesting 13 shares at $150 (~$1950 each).
        # First buy: available=2000 → max_qty=13, cost=1950 → approved, committed=1950.
        # Second buy: available=2000-1950=50 → max_qty=0 → rejected.
        # Without committed tracking both would be approved (total $3900 > $2000 available).
        portfolio = _base_portfolio(
            cash=7000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 150.0, "avg_cost": 0.0},
                {"ticker": "MSFT", "quantity": 0, "current_price": 150.0, "avg_cost": 0.0},
            ],
        )
        trade1 = _make_trade(ticker="AAPL", action="buy", quantity=13, confidence=0.9)
        trade2 = _make_trade(ticker="MSFT", action="buy", quantity=13, confidence=0.9)
        current_prices = {"AAPL": 150.0, "MSFT": 150.0}
        result = self.rm.validate_trades([trade1, trade2], portfolio, current_prices=current_prices)
        # Only first buy approved; second rejected because committed cash exhausted the budget
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ticker, "AAPL")

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
        from config.settings import MAX_POSITION_PCT
        # total_value=100000; buy 350 shares at $110
        # max_qty_by_position = floor(100000 * MAX_POSITION_PCT / 110)
        # cash=90000 → available=85000 → max_qty_by_cash=772 (not limiting)
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": 0, "current_price": 110.0, "avg_cost": 0.0}
            ],
        )
        trade = _make_trade(ticker="AAPL", action="buy", quantity=350, confidence=0.9)
        result = self.rm.validate_trades([trade], portfolio)
        expected_qty = int(100000.0 * MAX_POSITION_PCT / 110.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].quantity, expected_qty)

    def test_max_position_size_fully_at_cap_rejected(self):
        from config.settings import MAX_POSITION_PCT
        # Existing position already at cap → room=0 → rejected
        at_cap_qty = int(100000.0 * MAX_POSITION_PCT / 110.0) + 1
        portfolio = _base_portfolio(
            cash=70000.0,
            total_value=100000.0,
            positions=[
                {"ticker": "AAPL", "quantity": at_cap_qty, "current_price": 110.0, "avg_cost": 110.0}
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
        from config.settings import MAX_POSITIONS
        positions = [
            {"ticker": f"TICK{i}", "quantity": 1, "current_price": 100.0, "avg_cost": 100.0}
            for i in range(MAX_POSITIONS)
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
        from config.settings import MAX_POSITIONS
        positions = [
            {"ticker": f"TICK{i}", "quantity": 1, "current_price": 100.0, "avg_cost": 100.0}
            for i in range(MAX_POSITIONS)
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


    # Issue 15: Short entries blocked by daily loss circuit breaker
    @patch(f"{_RM_MODULE}.MAX_DAILY_LOSS_PCT", 0.03)
    def test_daily_loss_blocks_short_entry(self):
        """A sell that opens a new short should be blocked by daily loss breaker."""
        portfolio = _base_portfolio(
            total_value=100000.0,
            daily_pnl=-3500.0,  # 3.5% loss > 3% threshold
            positions=[],
        )
        short_trade = _make_trade(action="sell", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([short_trade], portfolio)
        self.assertEqual(result, [])

    @patch(f"{_RM_MODULE}.MAX_DAILY_LOSS_PCT", 0.03)
    def test_daily_loss_allows_normal_sell(self):
        """A sell that closes an existing long should pass even with daily loss breaker."""
        portfolio = _base_portfolio(
            total_value=100000.0,
            daily_pnl=-3500.0,
            positions=[{"ticker": "AAPL", "quantity": 50, "avg_cost": 150.0, "current_price": 150.0}],
        )
        sell_trade = _make_trade(action="sell", quantity=10, confidence=0.1)
        result = self.rm.validate_trades([sell_trade], portfolio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].action, "sell")

    # Issue 16: Short entries blocked by earnings blackout
    def test_earnings_blackout_blocks_short_entry(self):
        """A sell that opens a short during earnings blackout should be blocked."""
        today = date.today()
        earnings_date = today + timedelta(days=1)
        fundamentals = {"AAPL": {"next_earnings_date": earnings_date.isoformat()}}
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[],
        )
        short_trade = _make_trade(ticker="AAPL", action="sell", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([short_trade], portfolio, fundamentals=fundamentals)
        self.assertEqual(result, [])

    def test_earnings_blackout_allows_normal_sell(self):
        """A sell closing an existing long during earnings blackout should pass."""
        today = date.today()
        earnings_date = today + timedelta(days=1)
        fundamentals = {"AAPL": {"next_earnings_date": earnings_date.isoformat()}}
        portfolio = _base_portfolio(
            cash=90000.0,
            total_value=100000.0,
            positions=[{"ticker": "AAPL", "quantity": 50, "avg_cost": 150.0, "current_price": 150.0}],
        )
        sell_trade = _make_trade(ticker="AAPL", action="sell", quantity=10, confidence=0.9)
        result = self.rm.validate_trades([sell_trade], portfolio, fundamentals=fundamentals)
        self.assertEqual(len(result), 1)


class TestTrailingStopLoss(unittest.TestCase):
    def setUp(self):
        with patch("agent.risk_manager.get_trades", return_value=[]), \
             patch("agent.risk_manager.get_water_marks", return_value={}):
            self.rm = RiskManager()
        self._peak_patcher = patch("agent.risk_manager.get_max_equity_since", return_value=None)
        self._peak_patcher.start()
        self._upsert_wm_patcher = patch("agent.risk_manager.upsert_water_mark")
        self._upsert_wm_patcher.start()
        self._delete_wm_patcher = patch("agent.risk_manager.delete_water_mark")
        self._delete_wm_patcher.start()

    def tearDown(self):
        self._peak_patcher.stop()
        self._upsert_wm_patcher.stop()
        self._delete_wm_patcher.stop()

    def test_trailing_stop_long_triggers(self):
        """Long position: price ran up then dropped 12%+ from peak triggers trailing stop."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 100.0}],
        )
        # First call: price at 200 (sets high-water mark)
        self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 200.0})
        self.assertEqual(self.rm._high_water_marks["AAPL"], 200.0)

        # Second call: price drops to 175 (12.5% below 200 peak) -> trailing stop triggers
        forced = self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 175.0})
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0].action, "sell")
        self.assertIn("Trailing stop", forced[0].reasoning)

    def test_trailing_stop_long_no_trigger_below_threshold(self):
        """Long position: 10% drop from peak should NOT trigger trailing stop (threshold is 12%)."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 100.0}],
        )
        self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 200.0})
        # Price at 181 = 9.5% drop from 200 -> no trigger
        forced = self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 181.0})
        self.assertEqual(len(forced), 0)

    def test_trailing_stop_short_triggers(self):
        """Short position: price dropped then rose 12%+ from trough triggers trailing stop."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "TSLA", "quantity": -10, "avg_cost": 200.0, "current_price": 200.0}],
        )
        # First call: price at 150 (sets low-water mark)
        self.rm._check_stop_loss_take_profit(portfolio, {"TSLA": 150.0})
        self.assertEqual(self.rm._low_water_marks["TSLA"], 150.0)

        # Second call: price rises to 169 (12.67% above 150 trough) -> trailing stop
        forced = self.rm._check_stop_loss_take_profit(portfolio, {"TSLA": 169.0})
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0].action, "buy")
        self.assertIn("Trailing stop", forced[0].reasoning)

    def test_trailing_stop_short_no_trigger_below_threshold(self):
        """Short position: 10% rise from trough should NOT trigger trailing stop."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "TSLA", "quantity": -10, "avg_cost": 200.0, "current_price": 200.0}],
        )
        self.rm._check_stop_loss_take_profit(portfolio, {"TSLA": 150.0})
        # Price at 164 = 9.3% above 150 -> no trigger
        forced = self.rm._check_stop_loss_take_profit(portfolio, {"TSLA": 164.0})
        self.assertEqual(len(forced), 0)

    def test_hwm_initialized_from_entry_price(self):
        """High-water mark should initialize from avg_cost when first seen."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 95.0}],
        )
        # Price below avg_cost -> hwm should be avg_cost (100), not price (95)
        self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 95.0})
        self.assertEqual(self.rm._high_water_marks["AAPL"], 100.0)

    def test_water_marks_cleaned_on_position_close(self):
        """Water marks should be removed when a position disappears."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 150.0}],
        )
        self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 150.0})
        self.assertIn("AAPL", self.rm._high_water_marks)

        # Position closed
        empty_portfolio = _base_portfolio(positions=[])
        self.rm._check_stop_loss_take_profit(empty_portfolio, {})
        self.assertNotIn("AAPL", self.rm._high_water_marks)

    def test_fixed_stop_takes_priority_over_trailing(self):
        """When both fixed and trailing stop trigger, fixed stop reason is used."""
        portfolio = _base_portfolio(
            positions=[{"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 100.0}],
        )
        # Set hwm to 100 (entry price), then price drops to 87 (13% from entry, 13% from peak)
        # Both fixed (8% from entry) and trailing (12% from peak) trigger
        # Fixed should take priority
        forced = self.rm._check_stop_loss_take_profit(portfolio, {"AAPL": 87.0})
        self.assertEqual(len(forced), 1)
        self.assertIn("Stop-loss", forced[0].reasoning)
        self.assertNotIn("Trailing", forced[0].reasoning)


if __name__ == "__main__":
    unittest.main()
