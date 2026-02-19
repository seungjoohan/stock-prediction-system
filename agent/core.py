import logging
import threading
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo

from config.settings import (
    TRACKED_SYMBOLS,
    DECISION_INTERVAL_MIN,
    NEWS_POLL_INTERVAL_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)
from db.database import init_db, insert_agent_log, get_company_fundamentals, upsert_live_prices
from services.market_data import MarketDataService
from services.news_ingestion import fetch_latest_news
from services.fundamentals import FundamentalsService
from services.macro_data import MacroDataService
from services.llm_provider import LLMProvider
from services.trade_executor import TradeExecutor
from services.portfolio import PortfolioService
from agent.decision_engine import DecisionEngine
from agent.risk_manager import RiskManager
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


class AgentCore:
    def __init__(self):
        init_db()
        logger.info("Database initialized")

        self._llm = LLMProvider()
        self._market_data = MarketDataService()
        self._fundamentals = FundamentalsService()
        self._macro = MacroDataService()
        self._executor = TradeExecutor()
        self._portfolio = PortfolioService()
        self._decision_engine = DecisionEngine(self._llm)
        self._risk_manager = RiskManager()
        self._rag = RAGService()

        self.recent_sentiment: list[dict] = []
        self._minutes_elapsed = 0
        self._last_setup_date: date | None = None
        self._price_flush_running = False
        self._price_flush_thread: threading.Thread | None = None

        self._market_data.on_significant_move(self._on_significant_move)

        logger.info("AgentCore initialized with %d tracked symbols", len(TRACKED_SYMBOLS))

    def run(self):
        try:
            self._pre_market_setup()
            self._market_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt")
        finally:
            self._price_flush_running = False
            self._market_data.stop()
            logger.info("Market data service stopped; agent exiting")

    def _is_market_hours(self) -> bool:
        now = datetime.now(_ET)
        open_time = now.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        close_time = now.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return open_time <= now < close_time

    def _pre_market_setup(self):
        today = date.today()
        if self._last_setup_date == today:
            logger.info("Pre-market setup already done today (%s), skipping", today)
            return
        self._last_setup_date = today
        logger.info("Running pre-market setup for %s", today)

        try:
            self._fundamentals.refresh_all()
            logger.info("Company fundamentals refreshed for all tickers")
            insert_agent_log("fundamentals_refresh", "Pre-market fundamentals refresh complete")
        except Exception as exc:
            logger.warning("Failed to refresh fundamentals: %s", exc)
            insert_agent_log("fundamentals_refresh_error", str(exc))

        try:
            macro_snapshot = self._macro.refresh_macro_data()
            logger.info(
                "Macro data refreshed: vix=%.2f, fed_funds=%.2f",
                macro_snapshot.vix or 0.0,
                macro_snapshot.fed_funds_rate or 0.0,
            )
            insert_agent_log("macro_refresh", "Pre-market macro data refresh complete")
        except Exception as exc:
            logger.warning("Failed to refresh macro data: %s", exc)
            insert_agent_log("macro_refresh_error", str(exc))

        try:
            self._market_data.start()
            logger.info("Market data WebSocket started")
            insert_agent_log("market_data_start", "WebSocket connection established")
            self._start_price_flush_thread()
        except Exception as exc:
            logger.warning("Failed to start market data WebSocket: %s", exc)
            insert_agent_log("market_data_start_error", str(exc))

    def _start_price_flush_thread(self):
        """Flush in-memory WebSocket prices to SQLite every 5 seconds so the dashboard stays current."""
        if self._price_flush_thread and self._price_flush_thread.is_alive():
            return

        def _flush_loop():
            while self._price_flush_running:
                try:
                    prices = self._market_data.get_all_prices()
                    if prices:
                        upsert_live_prices(prices)
                except Exception as exc:
                    logger.debug("Price flush error (non-fatal): %s", exc)
                time.sleep(5)

        self._price_flush_running = True
        self._price_flush_thread = threading.Thread(
            target=_flush_loop,
            name="price-flush",
            daemon=True,
        )
        self._price_flush_thread.start()
        logger.info("Price flush thread started (interval=5s)")

    def _market_loop(self):
        logger.info("Entering market loop")
        was_market_hours = False

        while True:
            in_market = self._is_market_hours()

            # Detect market close: stop WebSocket to avoid reconnect churn overnight
            if not in_market and was_market_hours:
                logger.info("Market closed — stopping WebSocket until next open")
                self._market_data.stop()

            # Detect market open transition: re-run pre-market setup each new day
            if in_market and not was_market_hours:
                logger.info("Market just opened — running daily pre-market setup")
                self._pre_market_setup()
                self._minutes_elapsed = 0
            was_market_hours = in_market

            if not in_market:
                logger.debug("Outside market hours; sleeping 60s")
                time.sleep(60)
                continue

            if self._minutes_elapsed % NEWS_POLL_INTERVAL_MIN == 0:
                self._fetch_and_analyze_news()
                # Sync positions/prices from Alpaca on every news poll (every 5 min)
                # so the dashboard always reflects live prices between decision cycles
                try:
                    account = self._executor.get_account()
                    alpaca_positions = self._executor.get_positions()
                    self._portfolio.sync_from_alpaca(account, alpaca_positions)
                except Exception as exc:
                    logger.warning("Failed to sync portfolio from Alpaca: %s", exc)

            if self._minutes_elapsed % DECISION_INTERVAL_MIN == 0:
                self._run_decision_cycle()

                try:
                    prices = self._market_data.get_all_prices()
                    self._portfolio.take_snapshot(prices)
                    logger.info("Portfolio snapshot taken")
                except Exception as exc:
                    logger.warning("Failed to take portfolio snapshot: %s", exc)

            self._minutes_elapsed += 1
            time.sleep(60)

    def _fetch_and_analyze_news(self):
        try:
            news_items = fetch_latest_news(TRACKED_SYMBOLS)
            logger.info("Fetched %d new news items", len(news_items))
        except Exception as exc:
            logger.warning("Failed to fetch news: %s", exc)
            insert_agent_log("news_fetch_error", str(exc))
            return

        if not news_items:
            return

        # Cap at 25 most recent items (5 LLM batches) to stay within Groq 30 RPM limit
        if len(news_items) > 25:
            logger.info("Capping sentiment analysis at 25 of %d items", len(news_items))
            news_items = news_items[:25]

        try:
            sentiment_results = self._decision_engine.analyze_sentiment(news_items)
            self.recent_sentiment.extend(sentiment_results)
            if len(self.recent_sentiment) > 100:
                self.recent_sentiment = self.recent_sentiment[-100:]
            logger.info(
                "Sentiment analyzed: %d new signals, %d total retained",
                len(sentiment_results),
                len(self.recent_sentiment),
            )
            insert_agent_log("sentiment_analysis", {
                "new_signals": len(sentiment_results),
                "total_retained": len(self.recent_sentiment),
            })

            # Persist to SQL + ChromaDB for cross-session retrieval
            try:
                self._rag.index(news_items, sentiment_results)
            except Exception as rag_exc:
                logger.warning("RAG indexing failed (non-fatal): %s", rag_exc)
        except Exception as exc:
            logger.warning("Failed to analyze sentiment: %s", exc)
            insert_agent_log("sentiment_analysis_error", str(exc))

    def _run_decision_cycle(self):
        logger.info("Running decision cycle (minute=%d)", self._minutes_elapsed)

        try:
            prices = self._market_data.get_all_prices()
        except Exception as exc:
            logger.warning("Failed to get prices: %s", exc)
            return

        try:
            portfolio_state = self._portfolio.get_state(prices)
        except Exception as exc:
            logger.warning("Failed to get portfolio state: %s", exc)
            return

        fundamentals: dict = {}
        for ticker in TRACKED_SYMBOLS:
            row = get_company_fundamentals(ticker)
            if row is not None:
                fundamentals[ticker] = row

        macro_snapshot = self._macro.get_macro_snapshot()

        portfolio_state_dict = {
            "cash": portfolio_state.cash,
            "buying_power": portfolio_state.buying_power,
            "total_value": portfolio_state.total_value,
            "positions": portfolio_state.positions,
            "daily_pnl": portfolio_state.daily_pnl,
        }

        macro_snapshot_dict: dict | None = None
        if macro_snapshot is not None:
            macro_snapshot_dict = {
                "vix": macro_snapshot.vix,
                "fed_funds_rate": macro_snapshot.fed_funds_rate,
                "treasury_10y": macro_snapshot.treasury_10y,
                "treasury_2y": macro_snapshot.treasury_2y,
                "yield_curve_spread": macro_snapshot.yield_curve_spread,
                "cpi_yoy": macro_snapshot.cpi_yoy,
                "unemployment_rate": macro_snapshot.unemployment_rate,
                "gdp_growth_qoq": macro_snapshot.gdp_growth_qoq,
                "consumer_sentiment": macro_snapshot.consumer_sentiment,
                "initial_jobless_claims": macro_snapshot.initial_jobless_claims,
            }

        historical_context: dict = {}
        try:
            historical_context = self._rag.retrieve(list(prices.keys()))
        except Exception as exc:
            logger.warning("RAG retrieval failed (non-fatal): %s", exc)

        try:
            proposed_trades = self._decision_engine.make_decisions(
                portfolio_state=portfolio_state_dict,
                sentiment_signals=self.recent_sentiment,
                current_prices=prices,
                fundamentals=fundamentals,
                macro_snapshot=macro_snapshot_dict,
                historical_context=historical_context,
            )
            logger.info("Decision engine proposed %d trades", len(proposed_trades))
        except Exception as exc:
            logger.warning("Decision engine failed: %s", exc)
            insert_agent_log("decision_engine_error", str(exc))
            return

        try:
            approved_trades = self._risk_manager.validate_trades(
                trades=proposed_trades,
                portfolio_state=portfolio_state_dict,
                macro_snapshot=macro_snapshot_dict,
                fundamentals=fundamentals,
                current_prices=prices,
            )
            logger.info(
                "%d of %d proposed trades approved by risk manager",
                len(approved_trades),
                len(proposed_trades),
            )
        except Exception as exc:
            logger.warning("Risk manager failed: %s", exc)
            insert_agent_log("risk_manager_error", str(exc))
            return

        self._execute_trades(approved_trades)

        # Sync from Alpaca immediately after trades so next decision cycle
        # sees the updated positions and cash
        if approved_trades:
            try:
                account = self._executor.get_account()
                alpaca_positions = self._executor.get_positions()
                self._portfolio.sync_from_alpaca(account, alpaca_positions)
            except Exception as exc:
                logger.warning("Failed to sync portfolio after trades: %s", exc)

    def _execute_trades(self, approved_trades: list):
        for trade in approved_trades:
            try:
                result = self._executor.execute_trade(
                    ticker=trade.ticker,
                    action=trade.action,
                    quantity=trade.quantity,
                    reasoning=trade.reasoning,
                    confidence=trade.confidence,
                )
                if result is None:
                    logger.warning(
                        "Trade execution returned None for %s %s; skipping portfolio record",
                        trade.action,
                        trade.ticker,
                    )
                    continue
                filled_price = result.get("price", 0.0)
                if not filled_price:
                    logger.warning(
                        "No filled price for %s %s; skipping portfolio record",
                        trade.action,
                        trade.ticker,
                    )
                    continue

                if trade.action == "buy":
                    self._portfolio.record_buy(trade.ticker, trade.quantity, filled_price)
                elif trade.action == "sell":
                    self._portfolio.record_sell(trade.ticker, trade.quantity, filled_price)

                logger.info(
                    "Trade executed: %s %d %s @ %.2f (confidence=%.2f)",
                    trade.action.upper(),
                    trade.quantity,
                    trade.ticker,
                    filled_price,
                    trade.confidence,
                )
                insert_agent_log(
                    "trade_executed",
                    {
                        "action": trade.action,
                        "quantity": trade.quantity,
                        "filled_price": filled_price,
                        "confidence": trade.confidence,
                        "reasoning": trade.reasoning,
                    },
                    ticker=trade.ticker,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to execute trade %s %s: %s",
                    trade.action,
                    trade.ticker,
                    exc,
                )
                insert_agent_log(
                    "trade_execution_error",
                    {"action": trade.action, "ticker": trade.ticker, "error": str(exc)},
                    ticker=trade.ticker,
                )

    def _on_significant_move(self, symbol: str, change: float, price: float):
        logger.info(
            "Significant price move: %s changed %.2f%% to %.2f",
            symbol,
            change * 100,
            price,
        )
        insert_agent_log(
            "significant_move",
            {"change_pct": change, "price": price},
            ticker=symbol,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    agent = AgentCore()
    agent.run()
