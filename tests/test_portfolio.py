import unittest
from unittest.mock import patch, MagicMock


# Patch DB calls before the module is imported so PortfolioService.__init__
# does not execute real DB I/O at import or construction time.
_DB_MODULE = "services.portfolio"


class TestPortfolioServiceInitialState(unittest.TestCase):
    @patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value=None)
    def test_initial_state_no_snapshot(self, mock_snapshot):
        from services.portfolio import PortfolioService
        svc = PortfolioService()
        self.assertEqual(svc.cash, 100000.0)

    @patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value={"cash": 75000.0, "total_value": 90000.0, "realized_pnl": 500.0})
    def test_initial_state_loads_from_snapshot(self, mock_snapshot):
        from services.portfolio import PortfolioService
        svc = PortfolioService()
        self.assertEqual(svc.cash, 75000.0)
        self.assertEqual(svc.realized_pnl, 500.0)


class TestPortfolioServiceRecordBuy(unittest.TestCase):
    def _make_service(self, cash=100000.0, positions=None, snapshot=None):
        from services.portfolio import PortfolioService
        with patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value=snapshot):
            svc = PortfolioService()
        svc.cash = cash
        return svc

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_record_buy_updates_cash(self, mock_get_pos, mock_update):
        svc = self._make_service(cash=100000.0)
        svc.record_buy("AAPL", 10, 150.0)
        # cost = 10 * 150 = 1500
        self.assertAlmostEqual(svc.cash, 98500.0)

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_record_buy_updates_position_new(self, mock_get_pos, mock_update):
        svc = self._make_service(cash=100000.0)
        svc.record_buy("AAPL", 10, 150.0)
        mock_update.assert_called_once_with("AAPL", 10, 150.0)

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[
        {"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}
    ])
    def test_record_buy_updates_position_existing_avg_cost(self, mock_get_pos, mock_update):
        svc = self._make_service(cash=100000.0)
        # Existing: 10 shares @ 100; buying 10 more @ 150
        # new_avg = (10*100 + 10*150) / 20 = 2500/20 = 125.0
        svc.record_buy("AAPL", 10, 150.0)
        mock_update.assert_called_once_with("AAPL", 20, 125.0)

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_insufficient_cash_buy_rejected(self, mock_get_pos, mock_update):
        svc = self._make_service(cash=100.0)
        svc.record_buy("AAPL", 10, 150.0)
        # cost = 1500 > cash = 100 → rejected; cash unchanged, update not called
        self.assertAlmostEqual(svc.cash, 100.0)
        mock_update.assert_not_called()

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_insufficient_cash_does_not_raise(self, mock_get_pos, mock_update):
        svc = self._make_service(cash=0.0)
        try:
            svc.record_buy("AAPL", 1, 50.0)
        except Exception as exc:
            self.fail(f"record_buy raised unexpectedly: {exc}")


class TestPortfolioServiceRecordSell(unittest.TestCase):
    def _make_service_with_position(self, cash=50000.0, qty=10, avg_cost=100.0):
        from services.portfolio import PortfolioService
        with patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value=None):
            svc = PortfolioService()
        svc.cash = cash
        return svc

    def _patch_positions(self, qty=10, avg_cost=100.0):
        return patch(
            f"{_DB_MODULE}.get_positions",
            return_value=[{"ticker": "AAPL", "quantity": qty, "avg_cost": avg_cost}],
        )

    @patch(f"{_DB_MODULE}.update_position")
    def test_record_sell_updates_cash(self, mock_update):
        with self._patch_positions(qty=10, avg_cost=100.0):
            svc = self._make_service_with_position(cash=50000.0)
            svc.record_sell("AAPL", 5, 120.0)
        # proceeds = 5 * 120 = 600
        self.assertAlmostEqual(svc.cash, 50600.0)

    @patch(f"{_DB_MODULE}.update_position")
    def test_record_sell_computes_realized_pnl(self, mock_update):
        with self._patch_positions(qty=10, avg_cost=100.0):
            svc = self._make_service_with_position()
            svc.realized_pnl = 0.0
            svc.record_sell("AAPL", 5, 120.0)
        # pnl = (120 - 100) * 5 = 100
        self.assertAlmostEqual(svc.realized_pnl, 100.0)

    @patch(f"{_DB_MODULE}.update_position")
    def test_record_sell_caps_at_held_quantity(self, mock_update):
        with self._patch_positions(qty=5, avg_cost=100.0):
            svc = self._make_service_with_position(cash=50000.0)
            svc.record_sell("AAPL", 100, 120.0)
        # capped at 5 shares; update called with new_qty=0
        mock_update.assert_called_once_with("AAPL", 0, 100.0)
        # proceeds = 5 * 120 = 600
        self.assertAlmostEqual(svc.cash, 50600.0)

    @patch(f"{_DB_MODULE}.update_position")
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_record_sell_no_position_does_not_raise(self, mock_get_pos, mock_update):
        from services.portfolio import PortfolioService
        with patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value=None):
            svc = PortfolioService()
        try:
            svc.record_sell("AAPL", 5, 120.0)
        except Exception as exc:
            self.fail(f"record_sell raised unexpectedly: {exc}")
        mock_update.assert_not_called()

    @patch(f"{_DB_MODULE}.update_position")
    def test_record_sell_updates_position_quantity(self, mock_update):
        with self._patch_positions(qty=10, avg_cost=100.0):
            svc = self._make_service_with_position()
            svc.record_sell("AAPL", 3, 110.0)
        # new_quantity = 10 - 3 = 7; avg_cost preserved
        mock_update.assert_called_once_with("AAPL", 7, 100.0)


class TestPortfolioServiceGetState(unittest.TestCase):
    @patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value={"cash": 80000.0, "total_value": 100000.0, "realized_pnl": 0.0})
    @patch(f"{_DB_MODULE}.get_positions", return_value=[
        {"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0, "current_price": 110.0}
    ])
    def test_get_state_computes_total_value(self, mock_positions, mock_snapshot):
        from services.portfolio import PortfolioService
        svc = PortfolioService()
        svc.cash = 80000.0
        state = svc.get_state(current_prices={"AAPL": 150.0})
        # positions_value = 10 * 150 = 1500; total = 80000 + 1500 = 81500
        self.assertAlmostEqual(state.total_value, 81500.0)
        self.assertAlmostEqual(state.positions_value, 1500.0)

    @patch(f"{_DB_MODULE}.get_latest_portfolio_snapshot", return_value=None)
    @patch(f"{_DB_MODULE}.get_positions", return_value=[])
    def test_get_state_no_positions(self, mock_positions, mock_snapshot):
        from services.portfolio import PortfolioService
        svc = PortfolioService()
        svc.cash = 100000.0
        state = svc.get_state(current_prices={})
        self.assertAlmostEqual(state.cash, 100000.0)
        self.assertAlmostEqual(state.positions_value, 0.0)
        self.assertEqual(state.positions, [])


if __name__ == "__main__":
    unittest.main()
