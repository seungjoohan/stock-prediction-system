"""Two-layer RAG service for persistent news/sentiment memory.

Layer 1 — SQL: structured retrieval by ticker + time window.
Layer 2 — ChromaDB: semantic similarity search over past headlines.
"""
import hashlib
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

from db.database import insert_sentiment_signal, get_sentiment_signals
from services.news_ingestion import NewsItem

logger = logging.getLogger(__name__)

_CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "chroma")


class RAGService:
    def __init__(self) -> None:
        self._chroma_ok = False
        self._collection = None
        self._init_chroma()
        logger.info("RAGService initialized (chroma_available=%s)", self._chroma_ok)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            os.makedirs(_CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=_CHROMA_DIR)
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = client.get_or_create_collection(
                name="news_items",
                embedding_function=embed_fn,
            )
            self._chroma_ok = True
        except Exception as exc:
            logger.warning(
                "ChromaDB/sentence-transformers unavailable — RAG will use SQL only: %s", exc
            )
            self._chroma_ok = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, news_items: list[NewsItem], sentiment_signals: list[dict]) -> None:
        """Persist sentiment signals to SQL and news embeddings to ChromaDB."""
        # Layer 1: SQL persistence
        for signal in sentiment_signals:
            try:
                insert_sentiment_signal(signal)
            except Exception as exc:
                logger.warning("Failed to insert sentiment signal: %s", exc)

        # Layer 2: ChromaDB vector indexing
        if self._chroma_ok and self._collection is not None:
            self._index_chroma(news_items, sentiment_signals)

    def retrieve(self, tickers: list[str], days: int = 7, top_k: int = 3) -> dict:
        """Retrieve historical context for a list of tickers.

        Returns:
            {ticker: {"recent_signals": [...], "similar_news": [...]}}
        """
        result: dict = {}
        for ticker in tickers:
            recent_signals: list[dict] = []
            similar_news: list[dict] = []

            # Layer 1: SQL
            try:
                recent_signals = get_sentiment_signals(ticker, days=days)
            except Exception as exc:
                logger.warning("SQL retrieval failed for %s: %s", ticker, exc)

            # Layer 2: ChromaDB (skip if unavailable)
            if self._chroma_ok and self._collection is not None:
                query_text = next(
                    (s["news_headline"] for s in recent_signals if s.get("news_headline")),
                    None
                )

                if query_text is None:
                    logger.debug("No recent headline for %s — skipping ChromaDB semantic query", ticker)
                else:
                    try:
                        chroma_result = self._collection.query(
                            query_texts=[query_text],
                            n_results=min(top_k, self._collection.count()),
                            where={"ticker": ticker},
                        )
                        docs = chroma_result.get("documents", [[]])[0]
                        metas = chroma_result.get("metadatas", [[]])[0]
                        for doc, meta in zip(docs, metas):
                            similar_news.append({
                                "text": doc,
                                "timestamp": meta.get("timestamp", ""),
                                "sentiment": meta.get("sentiment"),
                                "confidence": meta.get("confidence"),
                            })
                    except Exception as exc:
                        logger.warning("ChromaDB query failed for %s: %s", ticker, exc)

            if recent_signals or similar_news:
                result[ticker] = {
                    "recent_signals": recent_signals,
                    "similar_news": similar_news,
                }

        logger.info("Historical context retrieved for %d tickers", len(result))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _index_chroma(self, news_items: list[NewsItem], sentiment_signals: list[dict]) -> None:
        """Upsert news item embeddings into ChromaDB."""
        # Build a fast lookup: ticker -> highest-confidence signal metadata
        signals_by_ticker: dict[str, list[dict]] = defaultdict(list)
        for sig in sentiment_signals:
            if ticker_key := sig.get("ticker", ""):
                signals_by_ticker[ticker_key].append(sig)

        signal_by_ticker: dict[str, dict] = {
            t: max(sigs, key=lambda s: s.get("confidence", 0.0))
            for t, sigs in signals_by_ticker.items()
        }

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for item in news_items:
            # Deterministic ID from headline content
            doc_id = hashlib.sha256(item.headline.encode()).hexdigest()[:32]

            text = item.headline
            if item.summary:
                text = f"{item.headline}. {item.summary}"

            timestamp = (
                item.timestamp.isoformat()
                if isinstance(item.timestamp, datetime)
                else str(item.timestamp)
            )

            for ticker in (item.tickers or []):
                sig = signal_by_ticker.get(ticker, {})
                # Make doc_id unique per ticker when a headline covers multiple tickers
                ticker_doc_id = f"{doc_id}_{ticker}"
                ids.append(ticker_doc_id)
                documents.append(text)
                metadatas.append({
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "sentiment": float(sig.get("sentiment", 0.0)),
                    "confidence": float(sig.get("confidence", 0.0)),
                })

        if not ids:
            return

        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.debug("Upserted %d documents to ChromaDB", len(ids))
        except Exception as exc:
            logger.warning("ChromaDB upsert failed: %s", exc)
