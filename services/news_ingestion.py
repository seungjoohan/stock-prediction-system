import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import feedparser
import finnhub

from config.settings import FINNHUB_API_KEY, RSS_FEEDS, TRACKED_SYMBOLS
from db.database import insert_news_item, is_news_duplicate

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    id: str
    source: str
    timestamp: datetime
    headline: str
    summary: str
    tickers: list[str]
    url: str
    raw_sentiment: float


class FinnhubNewsSource:
    def __init__(self) -> None:
        self._client = finnhub.Client(api_key=FINNHUB_API_KEY)

    def fetch(self, tickers: list[str]) -> list[NewsItem]:
        items: list[NewsItem] = []
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for symbol in tickers:
            try:
                articles = self._client.company_news(symbol, _from=date_str, to=date_str)
                for article in articles:
                    headline = article.get("headline", "").strip()
                    if not headline:
                        continue

                    ts_raw = article.get("datetime", 0)
                    try:
                        timestamp = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                    except (OSError, OverflowError, ValueError):
                        timestamp = datetime.now(timezone.utc)

                    items.append(
                        NewsItem(
                            id=str(uuid4()),
                            source="finnhub",
                            timestamp=timestamp,
                            headline=headline,
                            summary=article.get("summary", ""),
                            tickers=[symbol],
                            url=article.get("url", ""),
                            raw_sentiment=0.0,
                        )
                    )
            except Exception:
                logger.exception("Finnhub error fetching news for %s", symbol)

        return items


class RSSNewsSource:
    def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        tracked_upper = [s.upper() for s in TRACKED_SYMBOLS]

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    headline = getattr(entry, "title", "").strip()
                    if not headline:
                        continue

                    headline_upper = headline.upper()
                    matched_tickers = [
                        sym for sym in TRACKED_SYMBOLS
                        if sym.upper() in headline_upper
                    ]

                    published = getattr(entry, "published_parsed", None)
                    if published:
                        try:
                            timestamp = datetime(*published[:6], tzinfo=timezone.utc)
                        except (TypeError, ValueError):
                            timestamp = datetime.now(timezone.utc)
                    else:
                        timestamp = datetime.now(timezone.utc)

                    items.append(
                        NewsItem(
                            id=str(uuid4()),
                            source="rss",
                            timestamp=timestamp,
                            headline=headline,
                            summary=getattr(entry, "summary", ""),
                            tickers=matched_tickers,
                            url=getattr(entry, "link", ""),
                            raw_sentiment=0.0,
                        )
                    )
            except Exception:
                logger.exception("RSS error fetching feed %s", feed_url)

        return items


def fetch_latest_news(tickers: list[str]) -> list[NewsItem]:
    finnhub_source = FinnhubNewsSource()
    rss_source = RSSNewsSource()

    candidates: list[NewsItem] = []

    try:
        candidates.extend(finnhub_source.fetch(tickers))
    except Exception:
        logger.exception("Unexpected error from FinnhubNewsSource")

    try:
        candidates.extend(rss_source.fetch())
    except Exception:
        logger.exception("Unexpected error from RSSNewsSource")

    new_items: list[NewsItem] = []
    for item in candidates:
        try:
            if is_news_duplicate(item.headline):
                continue

            insert_news_item(
                {
                    "source": item.source,
                    "timestamp": item.timestamp.isoformat(),
                    "headline": item.headline,
                    "summary": item.summary,
                    "tickers": ",".join(item.tickers),
                    "url": item.url,
                    "raw_sentiment": item.raw_sentiment,
                }
            )
            new_items.append(item)
            logger.debug("Stored new news item: %s", item.headline[:80])
        except Exception:
            logger.exception("Error storing news item: %s", item.headline[:80])

    logger.info("fetch_latest_news: %d new items from %d candidates", len(new_items), len(candidates))
    return new_items
