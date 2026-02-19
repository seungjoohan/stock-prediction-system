import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import DB_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(db_dir, exist_ok=True)

    with _get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS news_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT,
                timestamp   TEXT,
                headline    TEXT NOT NULL,
                summary     TEXT,
                tickers     TEXT,
                url         TEXT,
                raw_sentiment REAL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                action      TEXT NOT NULL CHECK(action IN ('buy','sell')),
                quantity    REAL NOT NULL,
                price       REAL NOT NULL,
                total_value REAL NOT NULL,
                reasoning   TEXT,
                confidence  REAL,
                order_id    TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                total_value      REAL NOT NULL,
                cash             REAL NOT NULL,
                positions_value  REAL NOT NULL,
                realized_pnl     REAL,
                unrealized_pnl   REAL,
                daily_pnl        REAL
            );

            CREATE TABLE IF NOT EXISTS company_fundamentals (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker               TEXT NOT NULL UNIQUE,
                updated_at           TEXT NOT NULL,
                pe_ttm               REAL,
                forward_pe           REAL,
                ps_ratio             REAL,
                pb_ratio             REAL,
                ev_ebitda            REAL,
                gross_margin         REAL,
                operating_margin     REAL,
                roe                  REAL,
                roa                  REAL,
                revenue_growth_yoy   REAL,
                eps_growth_yoy       REAL,
                last_eps_actual      REAL,
                last_eps_estimate    REAL,
                earnings_surprise_pct REAL,
                next_earnings_date   TEXT,
                analyst_buy          INTEGER,
                analyst_hold         INTEGER,
                analyst_sell         INTEGER,
                avg_price_target     REAL,
                dividend_yield       REAL
            );

            CREATE TABLE IF NOT EXISTS macro_snapshots (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT NOT NULL,
                fed_funds_rate        REAL,
                treasury_10y          REAL,
                treasury_2y           REAL,
                yield_curve_spread    REAL,
                cpi_yoy               REAL,
                unemployment_rate     REAL,
                gdp_growth_qoq        REAL,
                vix                   REAL,
                consumer_sentiment    REAL,
                initial_jobless_claims REAL
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                event_type     TEXT NOT NULL,
                content        TEXT,
                related_ticker TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT NOT NULL UNIQUE,
                quantity      REAL NOT NULL,
                avg_cost      REAL NOT NULL,
                current_price REAL,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                sentiment       REAL,
                confidence      REAL,
                impact_horizon  TEXT,
                relevance_score REAL,
                reasoning       TEXT,
                news_headline   TEXT,
                UNIQUE(ticker, news_headline)
            );

            CREATE INDEX IF NOT EXISTS idx_sent_sig_ticker_ts
                ON sentiment_signals(ticker, timestamp);
        """)


def insert_trade(trade_dict: dict[str, Any]) -> int:
    trade_dict.setdefault("timestamp", _now_iso())
    columns = ["timestamp", "ticker", "action", "quantity", "price",
                "total_value", "reasoning", "confidence", "order_id"]
    values = [trade_dict.get(col) for col in columns]
    sql = (
        "INSERT INTO trades (timestamp, ticker, action, quantity, price, "
        "total_value, reasoning, confidence, order_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def insert_news_item(news_dict: dict[str, Any]) -> int:
    news_dict.setdefault("created_at", _now_iso())
    columns = ["source", "timestamp", "headline", "summary", "tickers",
                "url", "raw_sentiment", "created_at"]
    values = [news_dict.get(col) for col in columns]
    sql = (
        "INSERT INTO news_items (source, timestamp, headline, summary, tickers, "
        "url, raw_sentiment, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def insert_portfolio_snapshot(snapshot_dict: dict[str, Any]) -> int:
    snapshot_dict.setdefault("timestamp", _now_iso())
    columns = ["timestamp", "total_value", "cash", "positions_value",
                "realized_pnl", "unrealized_pnl", "daily_pnl"]
    values = [snapshot_dict.get(col) for col in columns]
    sql = (
        "INSERT INTO portfolio_snapshots (timestamp, total_value, cash, positions_value, "
        "realized_pnl, unrealized_pnl, daily_pnl) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def insert_agent_log(event_type: str, content: Any, ticker: str | None = None) -> int:
    content_json = json.dumps(content) if not isinstance(content, str) else content
    sql = (
        "INSERT INTO agent_logs (timestamp, event_type, content, related_ticker) "
        "VALUES (?, ?, ?, ?)"
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, [_now_iso(), event_type, content_json, ticker])
        return cursor.lastrowid


def get_positions() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
        return [dict(row) for row in rows]


def update_position(
    ticker: str, quantity: float, avg_cost: float, current_price: float | None = None
) -> None:
    if quantity <= 0:
        with _get_connection() as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", [ticker])
        return

    now = _now_iso()
    if current_price is not None:
        sql = (
            "INSERT INTO positions (ticker, quantity, avg_cost, current_price, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "quantity = excluded.quantity, avg_cost = excluded.avg_cost, "
            "current_price = excluded.current_price, updated_at = excluded.updated_at"
        )
        with _get_connection() as conn:
            conn.execute(sql, [ticker, quantity, avg_cost, current_price, now])
    else:
        sql = (
            "INSERT INTO positions (ticker, quantity, avg_cost, current_price, updated_at) "
            "VALUES (?, ?, ?, NULL, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "quantity = excluded.quantity, avg_cost = excluded.avg_cost, updated_at = excluded.updated_at"
        )
        with _get_connection() as conn:
            conn.execute(sql, [ticker, quantity, avg_cost, now])


def get_latest_portfolio_snapshot() -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_day_open_snapshot(date_str: str) -> dict | None:
    """Get the first portfolio snapshot of a given day (YYYY-MM-DD), or the last snapshot before it."""
    with _get_connection() as conn:
        # Try first snapshot of today
        row = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
            [date_str],
        ).fetchone()
        if row:
            return dict(row)
        # Fallback: last snapshot before today (previous close)
        row = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            [date_str],
        ).fetchone()
        return dict(row) if row else None


def get_trades(ticker: str | None = None, limit: int = 50) -> list[dict]:
    with _get_connection() as conn:
        if ticker is not None:
            rows = conn.execute(
                "SELECT * FROM trades WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                [ticker, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [dict(row) for row in rows]


def upsert_company_fundamentals(fundamentals_dict: dict[str, Any]) -> None:
    fundamentals_dict["updated_at"] = _now_iso()
    columns = [
        "ticker", "updated_at", "pe_ttm", "forward_pe", "ps_ratio", "pb_ratio",
        "ev_ebitda", "gross_margin", "operating_margin", "roe", "roa",
        "revenue_growth_yoy", "eps_growth_yoy", "last_eps_actual", "last_eps_estimate",
        "earnings_surprise_pct", "next_earnings_date", "analyst_buy", "analyst_hold",
        "analyst_sell", "avg_price_target", "dividend_yield",
    ]
    values = [fundamentals_dict.get(col) for col in columns]
    placeholders = ", ".join("?" * len(columns))
    col_names = ", ".join(columns)
    update_clause = ", ".join(
        f"{col} = excluded.{col}"
        for col in columns
        if col != "ticker"
    )
    sql = (
        f"INSERT INTO company_fundamentals ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT(ticker) DO UPDATE SET {update_clause}"
    )
    with _get_connection() as conn:
        conn.execute(sql, values)


def get_company_fundamentals(ticker: str) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM company_fundamentals WHERE ticker = ?", [ticker]
        ).fetchone()
        return dict(row) if row else None


def insert_macro_snapshot(snapshot_dict: dict[str, Any]) -> int:
    snapshot_dict.setdefault("timestamp", _now_iso())
    columns = [
        "timestamp", "fed_funds_rate", "treasury_10y", "treasury_2y",
        "yield_curve_spread", "cpi_yoy", "unemployment_rate", "gdp_growth_qoq",
        "vix", "consumer_sentiment", "initial_jobless_claims",
    ]
    values = [snapshot_dict.get(col) for col in columns]
    placeholders = ", ".join("?" * len(columns))
    col_names = ", ".join(columns)
    sql = f"INSERT INTO macro_snapshots ({col_names}) VALUES ({placeholders})"
    with _get_connection() as conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def get_latest_macro_snapshot() -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM macro_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def is_news_duplicate(headline: str) -> bool:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM news_items WHERE headline = ? LIMIT 1", [headline]
        ).fetchone()
        return row is not None


def insert_sentiment_signal(signal_dict: dict[str, Any]) -> int:
    timestamp = signal_dict.get("timestamp") or _now_iso()
    columns = [
        "timestamp", "ticker", "sentiment", "confidence",
        "impact_horizon", "relevance_score", "reasoning", "news_headline",
    ]
    values = [timestamp] + [signal_dict.get(col) for col in columns[1:]]
    sql = (
        "INSERT OR IGNORE INTO sentiment_signals "
        "(timestamp, ticker, sentiment, confidence, impact_horizon, "
        "relevance_score, reasoning, news_headline) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def get_sentiment_signals(ticker: str, days: int = 7) -> list[dict]:
    """Return sentiment signals for *ticker* from the last *days* calendar days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sentiment_signals "
            "WHERE ticker = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC",
            [ticker, cutoff],
        ).fetchall()
        return [dict(row) for row in rows]
