import unittest
from unittest.mock import MagicMock, patch, PropertyMock


_DB_MODULE = "services.trade_executor"
_ALPACA_MODULE = "services.trade_executor.TradingClient"


def _make_mock_order(order_id="order-123", qty=10, filled_avg_price=150.0, status="accepted"):
    order = MagicMock()
    order.id = order_id
    order.qty = qty
    order.filled_avg_price = filled_avg_price
    order.status = status
    return order


def _make_mock_account(
    cash=50000.0,
    portfolio_value=100000.0,
    buying_power=50000.0,
    equity=100000.0,
    last_equity=99000.0,
    status="ACTIVE",
):
    account = MagicMock()
    account.cash = cash
    account.portfolio_value = portfolio_value
    account.buying_power = buying_power
    account.equity = equity
    account.last_equity = last_equity
    account.status = status
    return account


def _make_mock_position(
    symbol="AAPL",
    qty=10.0,
    avg_entry_price=140.0,
    current_price=150.0,
    market_value=1500.0,
    unrealized_pl=100.0,
    unrealized_plpc=0.07,
    side="long",
):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.avg_entry_price = avg_entry_price
    pos.current_price = current_price
    pos.market_value = market_value
    pos.unrealized_pl = unrealized_pl
    pos.unrealized_plpc = unrealized_plpc
    pos.side = side
    return pos


@patch(f"{_DB_MODULE}.insert_agent_log")
@patch(f"{_DB_MODULE}.insert_trade")
@patch(_ALPACA_MODULE)
class TestTradeExecutorExecuteTrade(unittest.TestCase):
    def _make_executor(self, mock_client_class):
        from services.trade_executor import TradeExecutor
        executor = TradeExecutor.__new__(TradeExecutor)
        executor.client = mock_client_class.return_value
        return executor

    def test_execute_buy_order(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        mock_order = _make_mock_order(order_id="buy-001", qty=5, filled_avg_price=200.0, status="accepted")
        executor.client.submit_order.return_value = mock_order

        result = executor.execute_trade("AAPL", "buy", 5, reasoning="momentum", confidence=0.8)

        self.assertIsNotNone(result)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["action"], "buy")
        self.assertEqual(result["quantity"], 5)
        self.assertEqual(result["id"], "buy-001")
        executor.client.submit_order.assert_called_once()

    def test_execute_sell_order(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        mock_order = _make_mock_order(order_id="sell-002", qty=3, filled_avg_price=180.0, status="accepted")
        executor.client.submit_order.return_value = mock_order

        result = executor.execute_trade("MSFT", "sell", 3)

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "sell")
        self.assertEqual(result["ticker"], "MSFT")
        executor.client.submit_order.assert_called_once()

    def test_invalid_action_returns_none(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)

        result = executor.execute_trade("AAPL", "hold", 10)

        self.assertIsNone(result)
        executor.client.submit_order.assert_not_called()

    def test_invalid_action_logs_error(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.execute_trade("AAPL", "hold", 10)
        mock_log.assert_called()

    def test_order_failure_returns_none(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.submit_order.side_effect = RuntimeError("connection refused")

        result = executor.execute_trade("TSLA", "buy", 2)

        self.assertIsNone(result)

    def test_order_failure_logs_error(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.submit_order.side_effect = RuntimeError("connection refused")

        executor.execute_trade("TSLA", "buy", 2)

        # insert_agent_log should be called with TRADE_ERROR event
        error_calls = [
            call for call in mock_log.call_args_list
            if call.args and call.args[0] == "TRADE_ERROR"
        ]
        self.assertTrue(len(error_calls) > 0)

    def test_order_failure_still_calls_insert_trade(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.submit_order.side_effect = RuntimeError("timeout")

        executor.execute_trade("NVDA", "buy", 1)

        mock_insert_trade.assert_called_once()

    def test_execute_buy_case_insensitive(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        mock_order = _make_mock_order()
        executor.client.submit_order.return_value = mock_order

        result = executor.execute_trade("AAPL", "BUY", 1)

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "buy")

    def test_execute_trade_logs_success(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        mock_order = _make_mock_order()
        executor.client.submit_order.return_value = mock_order

        executor.execute_trade("AAPL", "buy", 10)

        success_calls = [
            call for call in mock_log.call_args_list
            if call.args and call.args[0] == "TRADE_EXECUTED"
        ]
        self.assertTrue(len(success_calls) > 0)


@patch(f"{_DB_MODULE}.insert_agent_log")
@patch(f"{_DB_MODULE}.insert_trade")
@patch(_ALPACA_MODULE)
class TestTradeExecutorGetAccount(unittest.TestCase):
    def _make_executor(self, mock_client_class):
        from services.trade_executor import TradeExecutor
        executor = TradeExecutor.__new__(TradeExecutor)
        executor.client = mock_client_class.return_value
        return executor

    def test_get_account_returns_dict(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.get_account.return_value = _make_mock_account(
            cash=50000.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            equity=100000.0,
            last_equity=99000.0,
            status="ACTIVE",
        )

        result = executor.get_account()

        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["cash"], 50000.0)
        self.assertAlmostEqual(result["portfolio_value"], 100000.0)
        self.assertAlmostEqual(result["buying_power"], 50000.0)
        self.assertAlmostEqual(result["equity"], 100000.0)
        self.assertIn("status", result)

    def test_get_account_exception_returns_empty_dict(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.get_account.side_effect = RuntimeError("API error")

        result = executor.get_account()

        self.assertEqual(result, {})


@patch(f"{_DB_MODULE}.insert_agent_log")
@patch(f"{_DB_MODULE}.insert_trade")
@patch(_ALPACA_MODULE)
class TestTradeExecutorGetPositions(unittest.TestCase):
    def _make_executor(self, mock_client_class):
        from services.trade_executor import TradeExecutor
        executor = TradeExecutor.__new__(TradeExecutor)
        executor.client = mock_client_class.return_value
        return executor

    def test_get_positions_returns_list(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        mock_pos = _make_mock_position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=140.0,
            current_price=150.0,
            market_value=1500.0,
            unrealized_pl=100.0,
            unrealized_plpc=0.07,
            side="long",
        )
        executor.client.get_all_positions.return_value = [mock_pos]

        result = executor.get_positions()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")
        self.assertAlmostEqual(result[0]["quantity"], 10.0)
        self.assertAlmostEqual(result[0]["current_price"], 150.0)
        self.assertIn("avg_entry_price", result[0])
        self.assertIn("market_value", result[0])
        self.assertIn("unrealized_pl", result[0])
        self.assertIn("side", result[0])

    def test_get_positions_returns_empty_list_on_exception(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        executor.client.get_all_positions.side_effect = RuntimeError("API error")

        result = executor.get_positions()

        self.assertEqual(result, [])

    def test_get_positions_multiple(self, mock_client_class, mock_insert_trade, mock_log):
        executor = self._make_executor(mock_client_class)
        pos1 = _make_mock_position(symbol="AAPL")
        pos2 = _make_mock_position(symbol="MSFT")
        executor.client.get_all_positions.return_value = [pos1, pos2]

        result = executor.get_positions()

        self.assertEqual(len(result), 2)
        tickers = [p["ticker"] for p in result]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)


if __name__ == "__main__":
    unittest.main()
