import threading
import time
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import websocket

from config.settings import (
    FINNHUB_API_KEY,
    FINNHUB_WS_URL,
    TRACKED_SYMBOLS,
    SIGNIFICANT_MOVE_PCT,
)

logger = logging.getLogger(__name__)

# Prices older than this are considered stale and excluded from decision cycles.
_STALE_PRICE_SECONDS = 300  # 5 minutes


@dataclass
class PriceUpdate:
    symbol: str
    price: float
    volume: float
    timestamp: datetime


class MarketDataService:
    def __init__(self):
        self.prices: dict[str, float] = {}
        self.timestamps: dict[str, datetime] = {}
        self.price_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=900)
        )
        self._ws = None
        self._ws_thread = None
        self._running = False
        self._callbacks: list = []
        self._lock = threading.Lock()
        self._reconnect_delay = 1

    def start(self):
        """Start WebSocket connection in a background thread."""
        if self._running and self._ws_thread and self._ws_thread.is_alive():
            logger.info("MarketDataService already running, skipping start")
            return
        self._running = True
        self._reconnect_delay = 1
        self._ws_thread = threading.Thread(target=self._connect, daemon=True)
        self._ws_thread.start()
        logger.info("MarketDataService started")

    def stop(self):
        """Stop WebSocket connection and wait for thread to exit."""
        self._running = False
        # Read self._ws under lock to avoid racing with _connect()'s assignment.
        with self._lock:
            ws = self._ws
        if ws is not None:
            ws.close()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        self._ws_thread = None
        logger.info("MarketDataService stopped")

    def get_price(self, symbol: str) -> float | None:
        """Get latest price for a symbol. Returns None if price is stale."""
        with self._lock:
            price = self.prices.get(symbol)
            ts = self.timestamps.get(symbol)
        if price is None:
            return None
        if ts is not None:
            age = (datetime.now() - ts).total_seconds()
            if age > _STALE_PRICE_SECONDS:
                logger.debug("Stale price for %s (age=%.0fs)", symbol, age)
                return None
        return price

    def get_all_prices(self) -> dict[str, float]:
        """Get all current prices, excluding any that have gone stale."""
        now = datetime.now()
        with self._lock:
            result = {}
            for symbol, price in self.prices.items():
                ts = self.timestamps.get(symbol)
                if ts is not None:
                    age = (now - ts).total_seconds()
                    if age > _STALE_PRICE_SECONDS:
                        logger.debug(
                            "Dropping stale price for %s (age=%.0fs)", symbol, age
                        )
                        continue
                result[symbol] = price
        return result

    def get_price_change(self, symbol: str, minutes: int) -> float | None:
        """Get price change % over last N minutes. Returns None if insufficient data."""
        if minutes == 0:
            return 0.0
        with self._lock:
            history = self.price_history.get(symbol)
            if not history:
                return None
            samples_needed = minutes * 60
            if len(history) < samples_needed:
                return None
            past_price = history[-samples_needed][0]
            current_price = history[-1][0]
        if past_price == 0:
            return None
        return (current_price - past_price) / past_price

    def on_significant_move(self, callback):
        """Register callback for significant price movements (>2% in 5 min)."""
        self._callbacks.append(callback)

    def _connect(self):
        """Establish WebSocket connection with auto-reconnect."""
        while self._running:
            try:
                url = f"{FINNHUB_WS_URL}?token={FINNHUB_API_KEY}"
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                # Assign under lock so stop() sees a consistent reference.
                with self._lock:
                    self._ws = ws
                # ping_interval/ping_timeout detect silent TCP hangs;
                # without these, a dropped connection stalls run_forever() indefinitely.
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                logger.error("WebSocket connection error: %s", exc)
                # Apply backoff on hard failures too (not only on clean closes).
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)
            if self._running:
                logger.info(
                    "Reconnecting in %d seconds...", self._reconnect_delay
                )
                time.sleep(self._reconnect_delay)

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message from Finnhub."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            # Finnhub sends {"type":"error","msg":"..."} on rate-limit and auth failures.
            # Without this check the error frame is silently swallowed and prices freeze.
            if msg_type == "error":
                logger.warning(
                    "Finnhub error frame: %s — forcing reconnect",
                    data.get("msg", "(no message)"),
                )
                ws.close()
                return

            if msg_type != "trade":
                return

            for trade in data.get("data", []):
                symbol = trade.get("s")
                price = trade.get("p")
                volume = trade.get("v")
                timestamp_ms = trade.get("t")
                if symbol is None or price is None:
                    continue
                # Guard against missing timestamp field to avoid TypeError on division.
                if timestamp_ms is None:
                    timestamp = datetime.now()
                else:
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                update = PriceUpdate(
                    symbol=symbol,
                    price=float(price),
                    volume=float(volume) if volume is not None else 0.0,
                    timestamp=timestamp,
                )
                with self._lock:
                    self.prices[symbol] = update.price
                    self.timestamps[symbol] = update.timestamp
                    self.price_history[symbol].append(
                        (update.price, update.timestamp)
                    )
                self._check_significant_moves(symbol)
        except Exception as exc:
            logger.error("Error processing message: %s", exc)

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status, close_msg):
        """Handle WebSocket close -- reconnect with backoff."""
        logger.warning(
            "WebSocket closed (status=%s, msg=%s)", close_status, close_msg
        )
        if self._running:
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _on_open(self, ws):
        """Subscribe to all tracked symbols on connect."""
        self._reconnect_delay = 1
        for symbol in TRACKED_SYMBOLS:
            payload = json.dumps({"type": "subscribe", "symbol": symbol})
            ws.send(payload)
        logger.info(
            "Subscribed to %d symbols on WebSocket open", len(TRACKED_SYMBOLS)
        )

    def _check_significant_moves(self, symbol: str):
        """Check if any symbol moved >SIGNIFICANT_MOVE_PCT in 5 minutes."""
        change = self.get_price_change(symbol, minutes=5)
        if change is None:
            return
        if abs(change) >= SIGNIFICANT_MOVE_PCT:
            with self._lock:
                price = self.prices.get(symbol)
            logger.info(
                "Significant move detected: %s %.2f%%", symbol, change * 100
            )
            for callback in self._callbacks:
                try:
                    # Dispatch in a separate thread so SQLite writes in the callback
                    # don't stall the WebSocket receive loop.
                    threading.Thread(
                        target=callback,
                        args=(symbol, change, price),
                        daemon=True,
                    ).start()
                except Exception as exc:
                    logger.error(
                        "Callback error for significant move on %s: %s",
                        symbol,
                        exc,
                    )
