import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agent.decision_engine import DecisionEngine, _extract_json, _parse_json_list
from agent.risk_manager import TradeAction
from services.news_ingestion import NewsItem


def _make_news_item(headline="Test headline", source="rss", tickers=None):
    return NewsItem(
        id="test-id",
        source=source,
        timestamp=datetime.now(timezone.utc),
        headline=headline,
        summary="Test summary",
        tickers=tickers if tickers is not None else ["AAPL"],
        url="https://example.com",
        raw_sentiment=0.0,
    )


def _make_llm():
    return MagicMock()


class TestExtractJson(unittest.TestCase):
    def test_extract_json_with_fences(self):
        text = '```json\n[{"ticker": "AAPL"}]\n```'
        result = _extract_json(text)
        self.assertEqual(result, '[{"ticker": "AAPL"}]')

    def test_extract_json_with_plain_fences(self):
        text = '```\n[{"ticker": "MSFT"}]\n```'
        result = _extract_json(text)
        self.assertEqual(result, '[{"ticker": "MSFT"}]')

    def test_extract_json_plain_array(self):
        text = '[{"ticker": "GOOG"}]'
        result = _extract_json(text)
        self.assertEqual(result, '[{"ticker": "GOOG"}]')

    def test_extract_json_with_preamble(self):
        text = 'Here is the output:\n[{"ticker": "TSLA"}]'
        result = _extract_json(text)
        self.assertEqual(result, '[{"ticker": "TSLA"}]')

    def test_extract_json_object_fallback(self):
        text = '{"ticker": "AMZN"}'
        result = _extract_json(text)
        self.assertEqual(result, '{"ticker": "AMZN"}')

    def test_extract_json_no_json_returns_stripped(self):
        text = "  no json here  "
        result = _extract_json(text)
        self.assertEqual(result, "no json here")


class TestParseJsonList(unittest.TestCase):
    def test_parse_json_list_valid(self):
        raw = '[{"ticker": "AAPL", "action": "buy"}]'
        result = _parse_json_list(raw)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")

    def test_parse_json_list_invalid(self):
        result = _parse_json_list("garbage input {{{{")
        self.assertEqual(result, [])

    def test_parse_json_list_non_list_json(self):
        raw = '{"ticker": "AAPL"}'
        result = _parse_json_list(raw)
        self.assertEqual(result, [])

    def test_parse_json_list_fenced(self):
        raw = '```json\n[{"ticker": "NVDA"}]\n```'
        result = _parse_json_list(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "NVDA")

    def test_parse_json_list_empty_list(self):
        result = _parse_json_list("[]")
        self.assertEqual(result, [])


@patch("agent.decision_engine.insert_agent_log")
class TestDecisionEngineSentiment(unittest.TestCase):
    def _make_engine(self, llm_response_content="[]"):
        llm = _make_llm()
        # Sentiment uses call_fast; decisions use call
        llm.call_fast.return_value = {
            "content": llm_response_content,
            "provider": "test",
            "model": "test-fast-model",
        }
        return DecisionEngine(llm=llm), llm

    def test_analyze_sentiment_empty(self, mock_log):
        engine, llm = self._make_engine()
        result = engine.analyze_sentiment([])
        self.assertEqual(result, [])
        llm.call_fast.assert_not_called()

    def test_analyze_sentiment_batching(self, mock_log):
        engine, llm = self._make_engine("[]")
        news_items = [_make_news_item(headline=f"Headline {i}") for i in range(7)]
        engine.analyze_sentiment(news_items)
        # 7 items → batch of 5 + batch of 2 → 2 LLM calls
        self.assertEqual(llm.call_fast.call_count, 2)

    def test_analyze_sentiment_single_batch(self, mock_log):
        engine, llm = self._make_engine("[]")
        news_items = [_make_news_item(headline=f"Headline {i}") for i in range(3)]
        engine.analyze_sentiment(news_items)
        self.assertEqual(llm.call_fast.call_count, 1)

    def test_analyze_sentiment_exact_batch_boundary(self, mock_log):
        engine, llm = self._make_engine("[]")
        news_items = [_make_news_item(headline=f"Headline {i}") for i in range(5)]
        engine.analyze_sentiment(news_items)
        self.assertEqual(llm.call_fast.call_count, 1)

    def test_analyze_sentiment_returns_parsed_results(self, mock_log):
        sentiment_json = json.dumps([
            {"ticker": "AAPL", "sentiment": 0.8, "confidence": 0.9, "impact_horizon": "short", "relevance_score": 0.7, "reasoning": "positive"}
        ])
        engine, llm = self._make_engine(sentiment_json)
        news_items = [_make_news_item()]
        result = engine.analyze_sentiment(news_items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")

    def test_analyze_sentiment_logs_on_success(self, mock_log):
        engine, llm = self._make_engine("[]")
        engine.analyze_sentiment([_make_news_item()])
        mock_log.assert_called()

    def test_analyze_sentiment_handles_llm_exception(self, mock_log):
        llm = _make_llm()
        llm.call_fast.side_effect = RuntimeError("LLM unavailable")
        engine = DecisionEngine(llm=llm)
        news_items = [_make_news_item()]
        result = engine.analyze_sentiment(news_items)
        self.assertEqual(result, [])


@patch("agent.decision_engine.insert_agent_log")
class TestDecisionEngineMakeDecisions(unittest.TestCase):
    _PORTFOLIO = {
        "cash": 50000.0,
        "total_value": 100000.0,
        "positions": [],
        "daily_pnl": 0.0,
    }

    def _make_engine_with_response(self, content):
        llm = _make_llm()
        llm.call.return_value = {
            "content": content,
            "provider": "test",
            "model": "test-model",
        }
        return DecisionEngine(llm=llm)

    def test_make_decisions_returns_trade_actions(self, mock_log):
        valid_json = json.dumps([
            {
                "ticker": "AAPL",
                "action": "buy",
                "quantity": 10,
                "reasoning": "strong momentum",
                "confidence": 0.85,
                "urgency": "medium",
            }
        ])
        engine = self._make_engine_with_response(valid_json)
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={"AAPL": 150.0},
            fundamentals={},
            macro_snapshot=None,
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TradeAction)
        self.assertEqual(result[0].ticker, "AAPL")
        self.assertEqual(result[0].action, "buy")
        self.assertEqual(result[0].quantity, 10)
        self.assertAlmostEqual(result[0].confidence, 0.85)
        self.assertEqual(result[0].urgency, "medium")

    def test_make_decisions_handles_malformed(self, mock_log):
        mixed_json = json.dumps([
            {
                "ticker": "AAPL",
                "action": "buy",
                "quantity": 5,
                "confidence": 0.9,
                "urgency": "high",
                "reasoning": "valid",
            },
            {
                # missing required "ticker" key
                "action": "buy",
                "quantity": 3,
            },
            {
                "ticker": "MSFT",
                "action": "sell",
                "quantity": 2,
                "confidence": 0.75,
                "urgency": "low",
                "reasoning": "valid sell",
            },
        ])
        engine = self._make_engine_with_response(mixed_json)
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        tickers = [t.ticker for t in result]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertEqual(len(result), 2)

    def test_make_decisions_empty_json(self, mock_log):
        engine = self._make_engine_with_response("[]")
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        self.assertEqual(result, [])

    def test_make_decisions_garbage_response(self, mock_log):
        engine = self._make_engine_with_response("not json at all")
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        self.assertEqual(result, [])

    def test_make_decisions_llm_exception_returns_empty(self, mock_log):
        llm = _make_llm()
        llm.call.side_effect = RuntimeError("timeout")
        engine = DecisionEngine(llm=llm)
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        self.assertEqual(result, [])

    def test_make_decisions_logs_on_success(self, mock_log):
        valid_json = json.dumps([
            {
                "ticker": "NVDA",
                "action": "buy",
                "quantity": 1,
                "confidence": 0.7,
                "urgency": "low",
                "reasoning": "test",
            }
        ])
        engine = self._make_engine_with_response(valid_json)
        engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        mock_log.assert_called()

    def test_make_decisions_fenced_json(self, mock_log):
        fenced = '```json\n[{"ticker": "AMD", "action": "sell", "quantity": 3, "confidence": 0.6, "urgency": "low", "reasoning": "trim"}]\n```'
        engine = self._make_engine_with_response(fenced)
        result = engine.make_decisions(
            portfolio_state=self._PORTFOLIO,
            sentiment_signals=[],
            current_prices={},
            fundamentals={},
            macro_snapshot=None,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ticker, "AMD")


if __name__ == "__main__":
    unittest.main()
