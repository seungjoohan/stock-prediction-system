import logging
import os
import time
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from db.database import insert_trade, insert_agent_log, get_positions

logger = logging.getLogger(__name__)

_paper_mode = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"


class TradeExecutor:
    def __init__(self):
        logger.info("Trading mode: %s", "PAPER" if _paper_mode else "LIVE")
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=_paper_mode,
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
                "trade_error",
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

        # For sell trades, capture avg_cost_at_time before the position is mutated
        # so dashboard analytics always have a point-in-time cost basis.
        avg_cost_at_time = None
        if action_lower == "sell":
            try:
                positions = get_positions()
                pos = next((p for p in positions if p["ticker"] == ticker), None)
                if pos:
                    avg_cost_at_time = pos["avg_cost"]
            except Exception:
                pass

        trade_id = insert_trade({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "action": action_lower,
            "quantity": quantity,
            "price": price,
            "total_value": total_value,
            "reasoning": reasoning,
            "confidence": confidence,
            "order_id": order_id,
            "avg_cost_at_time": avg_cost_at_time,
        })

        if error_message is not None:
            insert_agent_log(
                "trade_error",
                {"ticker": ticker, "action": action_lower, "quantity": quantity, "error": error_message},
                ticker,
            )
            return None

        order_dict = {
            "id": order_id,
            "trade_id": trade_id,
            "ticker": ticker,
            "action": action_lower,
            "quantity": quantity,
            "price": price,
            "total_value": total_value,
            "status": str(order.status) if order is not None else "unknown",
        }

        insert_agent_log(
            "trade_executed",
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
        # Determine position side before closing so we know the correct action
        action = "sell"  # default for closing longs
        avg_cost_at_time = None
        try:
            positions = get_positions()
            pos = next((p for p in positions if p["ticker"] == ticker), None)
            if pos:
                avg_cost_at_time = pos["avg_cost"]
                if pos["quantity"] < 0:
                    action = "buy"  # buy-to-cover for shorts
        except Exception:
            pass

        try:
            order = self.client.close_position(ticker)
            order_id = str(order.id)

            # Wait briefly for fill price (paper orders fill near-instantly)
            for _ in range(6):
                if order.filled_avg_price is not None:
                    break
                time.sleep(0.5)
                try:
                    order = self.client.get_order_by_id(order_id)
                except Exception:
                    break

            filled_qty = float(order.filled_qty or order.qty or 0)
            filled_price = float(order.filled_avg_price or 0.0)

            order_dict = {
                "id": order_id,
                "ticker": ticker,
                "action": action,
                "quantity": filled_qty,
                "price": filled_price,
                "status": str(order.status),
            }

            insert_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "action": action,
                "quantity": filled_qty,
                "price": filled_price,
                "total_value": filled_qty * filled_price,
                "reasoning": "position closed (stop-loss/risk)",
                "confidence": 0.0,
                "order_id": order_id,
                "avg_cost_at_time": avg_cost_at_time,
            })

            insert_agent_log("position_closed", order_dict, ticker)
            logger.info("Closed position for %s, order_id=%s", ticker, order.id)
            return order_dict
        except Exception as exc:
            error_message = str(exc)
            logger.error("Failed to close position for %s: %s", ticker, error_message)
            insert_agent_log(
                "position_close_error",
                {"ticker": ticker, "error": error_message},
                ticker,
            )
            return None
