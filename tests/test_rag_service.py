"""Unit tests for RAGService.

Run with:  python -m pytest tests/test_rag_service.py -v
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure project root is on the path when running tests directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_news_item(headline: str, tickers: list[str] | None = None):
    """Create a minimal NewsItem-like object without importing the real class."""
    item = MagicMock()
    item.id = "test-id"
    item.headline = headline
    item.summary = "Test summary."
    item.tickers = tickers or ["AAPL"]
    item.timestamp = datetime(2026, 2, 18, 12, 0, tzinfo=timezone.utc)
    return item


def _make_signal(ticker: str, sentiment: float = 0.5, confidence: float = 0.8) -> dict:
    return {
        "ticker": ticker,
        "sentiment": sentiment,
        "confidence": confidence,
        "impact_horizon": "short_term",
        "relevance_score": 0.9,
        "reasoning": "Test reasoning",
        "news_headline": "Test headline",
    }


# ---------------------------------------------------------------------------
# Tests for _format_historical_context (no I/O required)
# ---------------------------------------------------------------------------

class TestFormatHistoricalContext(unittest.TestCase):
    def setUp(self):
        from agent.decision_engine import _format_historical_context
        self.fmt = _format_historical_context

    def test_none_returns_no_context(self):
        result = self.fmt(None)
        self.assertIn("No historical context", result)

    def test_empty_dict_returns_no_context(self):
        result = self.fmt({})
        self.assertIn("No historical context", result)

    def test_single_ticker_with_signals(self):
        ctx = {
            "NVDA": {
                "recent_signals": [
                    {"sentiment": 0.7, "confidence": 0.8},
                    {"sentiment": 0.5, "confidence": 0.7},
                ],
                "similar_news": [],
            }
        }
        result = self.fmt(ctx)
        self.assertIn("NVDA", result)
        self.assertIn("Sentiment trend", result)
        self.assertIn("bullish", result)

    def test_bearish_label(self):
        ctx = {
            "MPC": {
                "recent_signals": [{"sentiment": -0.5}, {"sentiment": -0.4}],
                "similar_news": [],
            }
        }
        result = self.fmt(ctx)
        self.assertIn("bearish", result)

    def test_similar_news_included(self):
        ctx = {
            "AAPL": {
                "recent_signals": [],
                "similar_news": [
                    {
                        "timestamp": "2026-02-12T10:00:00",
                        "text": "Apple beats earnings expectations",
                        "sentiment": 0.8,
                    }
                ],
            }
        }
        result = self.fmt(ctx)
        self.assertIn("Similar past news", result)
        self.assertIn("Apple beats earnings expectations", result)
        self.assertIn("+0.80", result)

    def test_all_tickers_included(self):
        ctx = {
            "AAPL": {"recent_signals": [{"sentiment": 0.5}], "similar_news": []},
            "MSFT": {"recent_signals": [{"sentiment": 0.3}], "similar_news": []},
            "GOOG": {"recent_signals": [{"sentiment": 0.1}], "similar_news": []},
        }
        result = self.fmt(ctx)
        # All tickers with data should appear (cap removed)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertIn("GOOG", result)

    def test_long_text_truncated(self):
        long_text = "A" * 200
        ctx = {
            "TSLA": {
                "recent_signals": [],
                "similar_news": [{"timestamp": "2026-02-10T00:00:00", "text": long_text, "sentiment": 0.1}],
            }
        }
        result = self.fmt(ctx)
        # Output line should be shorter than the raw text
        self.assertLess(len(result), 400)
        self.assertIn("...", result)


# ---------------------------------------------------------------------------
# Tests for RAGService (SQL layer only — ChromaDB mocked out)
# ---------------------------------------------------------------------------

class TestRAGServiceSQLLayer(unittest.TestCase):
    """Test RAGService index/retrieve using a real in-memory SQLite DB but
    with ChromaDB patched away so no embedding model download is needed."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "test_trading.db")

        import db.database as dbmod
        self._dbmod = dbmod
        self._original_db_path = dbmod.DB_PATH
        dbmod.DB_PATH = self._db_path
        self.addCleanup(setattr, dbmod, 'DB_PATH', self._original_db_path)  # guaranteed cleanup
        dbmod.init_db()

        self._chroma_patcher = patch.dict("sys.modules", {
            "chromadb": None,
            "chromadb.utils": None,
            "chromadb.utils.embedding_functions": None,
        })
        self._chroma_patcher.start()
        self.addCleanup(self._chroma_patcher.stop)  # guaranteed cleanup

        from services.rag_service import RAGService
        self._svc = RAGService()

    def tearDown(self):
        # addCleanup handles DB_PATH and chroma_patcher restoration
        pass

    def test_chroma_disabled_when_unavailable(self):
        self.assertFalse(self._svc._chroma_ok)

    def test_index_inserts_sentiment_signals(self):
        news = [_make_news_item("NVDA data center growth accelerates", ["NVDA"])]
        signals = [_make_signal("NVDA", sentiment=0.75)]

        self._svc.index(news, signals)

        rows = self._dbmod.get_sentiment_signals("NVDA", days=1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["sentiment"], 0.75)
        self.assertEqual(rows[0]["ticker"], "NVDA")

    def test_index_multiple_signals(self):
        news = []
        signals = [
            _make_signal("AAPL", sentiment=0.4),
            _make_signal("MSFT", sentiment=-0.2),
        ]
        self._svc.index(news, signals)

        aapl_rows = self._dbmod.get_sentiment_signals("AAPL", days=1)
        msft_rows = self._dbmod.get_sentiment_signals("MSFT", days=1)
        self.assertEqual(len(aapl_rows), 1)
        self.assertEqual(len(msft_rows), 1)

    def test_retrieve_returns_sql_signals(self):
        signals = [_make_signal("JPM", sentiment=0.3)]
        self._svc.index([], signals)

        result = self._svc.retrieve(["JPM"], days=7)
        self.assertIn("JPM", result)
        self.assertEqual(len(result["JPM"]["recent_signals"]), 1)
        self.assertEqual(result["JPM"]["similar_news"], [])  # no chroma

    def test_retrieve_unknown_ticker_returns_empty(self):
        result = self._svc.retrieve(["ZZZZ"], days=7)
        self.assertNotIn("ZZZZ", result)

    def test_retrieve_respects_days_window(self):
        """Signals older than the window should not be returned."""
        from datetime import timedelta
        import db.database as dbmod

        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat()
        dbmod.insert_sentiment_signal({
            "timestamp": old_ts,
            "ticker": "XOM",
            "sentiment": -0.5,
            "confidence": 0.7,
        })

        result = self._svc.retrieve(["XOM"], days=7)
        # 10-day old signal should be outside the 7-day window
        self.assertNotIn("XOM", result)


# ---------------------------------------------------------------------------
# Tests for database functions directly
# ---------------------------------------------------------------------------

class TestDatabaseSentimentFunctions(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "test_db.db")

        import db.database as dbmod
        self._dbmod = dbmod
        self._original_db_path = dbmod.DB_PATH
        dbmod.DB_PATH = self._db_path
        self.addCleanup(setattr, dbmod, 'DB_PATH', self._original_db_path)
        dbmod.init_db()

    def tearDown(self):
        pass  # addCleanup handles restoration

    def test_insert_and_retrieve(self):
        signal = {
            "ticker": "TSLA",
            "sentiment": -0.3,
            "confidence": 0.65,
            "impact_horizon": "medium_term",
            "relevance_score": 0.8,
            "reasoning": "EV demand slowing",
            "news_headline": "Tesla cuts prices again",
        }
        row_id = self._dbmod.insert_sentiment_signal(signal)
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

        rows = self._dbmod.get_sentiment_signals("TSLA", days=7)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "TSLA")
        self.assertAlmostEqual(rows[0]["sentiment"], -0.3)
        self.assertEqual(rows[0]["news_headline"], "Tesla cuts prices again")

    def test_get_signals_filters_by_ticker(self):
        self._dbmod.insert_sentiment_signal({"ticker": "AAPL", "sentiment": 0.5})
        self._dbmod.insert_sentiment_signal({"ticker": "GOOG", "sentiment": -0.2})

        aapl = self._dbmod.get_sentiment_signals("AAPL", days=1)
        goog = self._dbmod.get_sentiment_signals("GOOG", days=1)

        self.assertEqual(len(aapl), 1)
        self.assertEqual(len(goog), 1)
        self.assertEqual(aapl[0]["ticker"], "AAPL")

    def test_get_signals_empty_for_unknown_ticker(self):
        rows = self._dbmod.get_sentiment_signals("UNKNOWN", days=7)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
