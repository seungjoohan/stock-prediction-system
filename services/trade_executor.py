import logging
import time
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from db.database import insert_trade, insert_agent_log

logger = logging.getLogger(__name__)


class TradeExecutor:
    def __init__(self):
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True,
        )

    def execute_trade(self, ticker: str, action: str, quantity: int, reasoning: str = "", confidence: float = 0.0) -> dict | None:
        """Execute a market order via Alpaca.
        Returns order dict on success, None on failure.
        Logs to DB regardless of outcome.
        """
        action_lower = action.lower()
        if action_lower == "buy":
            side = OrderSide.BUY
        elif action_lower == "sell":
            side = OrderSide.SELL
        else:
            logger.error("Invalid action '%s' for ticker %s", action, ticker)
            insert_agent_log(
                "TRADE_ERROR",
                {"ticker": ticker, "action": action, "error": f"Invalid action: {action}"},
                ticker,
            )
            return None

        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        order = None
        order_id = None
        price = 0.0
        total_value = 0.0
        error_message = None

        try:
            order = self.client.submit_order(order_request)
            order_id = str(order.id)

            # Paper orders fill almost instantly — wait up to 3s for fill price
            for _ in range(6):
                if order.filled_avg_price is not None:
                    break
                time.sleep(0.5)
                try:
                    order = self.client.get_order_by_id(order_id)
                except Exception:
                    break

            filled_qty = float(order.filled_qty or order.qty or quantity)
            filled_price = float(order.filled_avg_price or 0.0)
            price = filled_price
            total_value = filled_qty * filled_price
            logger.info(
                "Order submitted: %s %s x%s @ %.2f, order_id=%s",
                action_lower,
                ticker,
                quantity,
                filled_price,
                order_id,
            )
        except Exception as exc:
            error_message = str(exc)
            logger.error(
                "Order submission failed for %s %s x%s: %s",
                action_lower,
                ticker,
                quantity,
                error_message,
            )

        insert_trade({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "action": action_lower,
            "quantity": quantity,
            "price": price,
            "total_value": total_value,
            "reasoning": reasoning,
            "confidence": confidence,
            "order_id": order_id,
        })

        if error_message is not None:
            insert_agent_log(
                "TRADE_ERROR",
                {"ticker": ticker, "action": action_lower, "quantity": quantity, "error": error_message},
                ticker,
            )
            return None

        order_dict = {
            "id": order_id,
            "ticker": ticker,
            "action": action_lower,
            "quantity": quantity,
            "price": price,
            "total_value": total_value,
            "status": str(order.status) if order is not None else "unknown",
        }

        insert_agent_log(
            "TRADE_EXECUTED",
            order_dict,
            ticker,
        )

        return order_dict

    def get_account(self) -> dict:
        """Get Alpaca account info (cash, portfolio value, etc.)."""
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "status": str(account.status),
            }
        except Exception as exc:
            logger.error("Failed to fetch account info: %s", exc)
            return {}

    def get_positions(self) -> list[dict]:
        """Get current Alpaca positions."""
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "ticker": pos.symbol,
                    "quantity": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "side": str(pos.side),
                }
                for pos in positions
            ]
        except Exception as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return []

    def close_position(self, ticker: str) -> dict | None:
        """Close entire position for a ticker."""
        try:
            order = self.client.close_position(ticker)
            order_dict = {
                "id": str(order.id),
                "ticker": ticker,
                "action": "sell",
                "status": str(order.status),
            }
            insert_agent_log("POSITION_CLOSED", order_dict, ticker)
            logger.info("Closed position for %s, order_id=%s", ticker, order.id)
            return order_dict
        except Exception as exc:
            error_message = str(exc)
            logger.error("Failed to close position for %s: %s", ticker, error_message)
            insert_agent_log(
                "POSITION_CLOSE_ERROR",
                {"ticker": ticker, "error": error_message},
                ticker,
            )
            return None
