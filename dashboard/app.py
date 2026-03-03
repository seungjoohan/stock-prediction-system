import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DB_PATH, INITIAL_CAPITAL, ALPACA_API_KEY, ALPACA_SECRET_KEY

st.set_page_config(page_title="Trading Agent Dashboard", layout="wide")

_ET = ZoneInfo("America/New_York")


def _to_et(ts_str, date_only: bool = False) -> str:
    """Convert a UTC ISO timestamp string to Eastern Time for display."""
    s = str(ts_str).strip() if ts_str else ""
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et = dt.astimezone(_ET)
        return et.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M ET")
    except (ValueError, TypeError):
        return s


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=30)
def fetch_latest_snapshot() -> dict | None:
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_portfolio_history() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, total_value, cash, positions_value, realized_pnl, unrealized_pnl, daily_pnl "
                "FROM portfolio_snapshots ORDER BY timestamp ASC",
                conn,
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_positions() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT ticker, quantity, avg_cost, current_price, updated_at FROM positions ORDER BY ticker",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_alpaca_account() -> dict:
    """Fetch live account + all positions (long & short) from Alpaca.
    Returns {} on any failure so callers can fall back to DB data.
    """
    try:
        import os
        from alpaca.trading.client import TradingClient as _TC
        _paper = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
        _client = _TC(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=_paper)
        acct = _client.get_account()
        positions = _client.get_all_positions()
        total_unrealized = sum(float(p.unrealized_pl) for p in positions)
        total_cost_basis = sum(float(p.cost_basis) for p in positions)
        long_mv  = sum(float(p.market_value) for p in positions if float(p.qty) > 0)
        short_mv = sum(float(p.market_value) for p in positions if float(p.qty) < 0)
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "long_market_value": long_mv,
            "short_market_value": short_mv,
            "unrealized_pnl": total_unrealized,
            "cost_basis": total_cost_basis,
            "n_long": sum(1 for p in positions if float(p.qty) > 0),
            "n_short": sum(1 for p in positions if float(p.qty) < 0),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_day_open_snapshot() -> dict | None:
    """First snapshot of today, used as the daily P&L baseline."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                [today_str],
            ).fetchone()
            if row:
                return dict(row)
            # Fallback: last snapshot before today (previous close)
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
                [today_str],
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


@st.cache_data(ttl=5)
def fetch_live_prices() -> pd.DataFrame:
    """Fetch real-time prices written by the agent's WebSocket flush thread (updated every 5s)."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT ticker, price AS live_price, updated_at AS price_updated_at FROM live_prices",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_trades(ticker_filter: str | None = None) -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            if ticker_filter and ticker_filter != "All":
                df = pd.read_sql_query(
                    "SELECT * FROM trades WHERE ticker = ? ORDER BY timestamp DESC",
                    conn,
                    params=(ticker_filter,),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp DESC",
                    conn,
                )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_all_tickers_traded() -> list[str]:
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM trades ORDER BY ticker"
            ).fetchall()
            return [r["ticker"] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=30)
def fetch_fundamentals() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT ticker, pe_ttm, forward_pe, gross_margin, operating_margin, roe, "
                "analyst_buy, analyst_hold, analyst_sell, avg_price_target, dividend_yield, updated_at "
                "FROM company_fundamentals ORDER BY ticker",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_latest_macro() -> dict | None:
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM macro_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_agent_logs(limit: int = 100) -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT id, timestamp, event_type, content, related_ticker "
                "FROM agent_logs ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_performance_stats() -> dict:
    stats = {
        "total_trades": 0,
        "buy_count": 0,
        "sell_count": 0,
        "win_rate": None,
        "best_trade": None,
        "worst_trade": None,
        "profit_factor": None,
        "avg_win": None,
        "avg_loss": None,
        "sell_count_evaluated": 0,
        "short_count_evaluated": 0,
    }
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
            stats["total_trades"] = row["cnt"] if row else 0

            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE action='buy'"
            ).fetchone()
            stats["buy_count"] = row["cnt"] if row else 0

            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE action='sell'"
            ).fetchone()
            stats["sell_count"] = row["cnt"] if row else 0

            # --- Long trades: sells matched to prior buys ---
            sell_rows = conn.execute(
                "SELECT ticker, price as sell_price, quantity, total_value, timestamp, avg_cost_at_time "
                "FROM trades "
                "WHERE action = 'sell' AND price > 0 AND quantity > 0",
            ).fetchall()

            long_pnls = []
            for r in sell_rows:
                avg_cost = r["avg_cost_at_time"]
                if avg_cost is None:
                    # Reconstruct point-in-time avg_cost from cumulative buy history
                    # for trades that predate the avg_cost_at_time column.
                    buy_agg = conn.execute(
                        "SELECT SUM(price * quantity) / SUM(quantity) as avg_cost "
                        "FROM trades "
                        "WHERE ticker = ? AND action = 'buy' "
                        "AND price > 0 AND quantity > 0 AND timestamp < ?",
                        [r["ticker"], r["timestamp"]],
                    ).fetchone()
                    if buy_agg and buy_agg["avg_cost"] is not None:
                        avg_cost = buy_agg["avg_cost"]
                if avg_cost is not None and avg_cost > 0:
                    pnl = (r["sell_price"] - avg_cost) * r["quantity"]
                    long_pnls.append({
                        "ticker": r["ticker"],
                        "timestamp": r["timestamp"],
                        "pnl": pnl,
                        "trade_type": "long",
                    })
            stats["sell_count_evaluated"] = len(long_pnls)

            # --- Short trades: FIFO position simulation ---
            # Simulate net position per ticker chronologically.
            # When a sell pushes position to ≤0 → short entry (FIFO queue).
            # When a buy reduces a negative position → short close, compute PnL.
            from collections import defaultdict, deque

            all_trades_for_sim = conn.execute(
                "SELECT ticker, action, price, quantity, timestamp "
                "FROM trades WHERE price > 0 AND quantity > 0 "
                "ORDER BY timestamp ASC",
            ).fetchall()

            net_positions: dict = defaultdict(float)
            short_queues: dict = defaultdict(deque)  # (entry_price, qty) per ticker
            short_pnls = []

            for t in all_trades_for_sim:
                ticker = t["ticker"]
                action = t["action"]
                price = float(t["price"])
                qty = float(t["quantity"])
                cur_pos = net_positions[ticker]

                if action == "sell":
                    if cur_pos >= qty:
                        # Closing/reducing a long — long-close logic handles PnL above
                        net_positions[ticker] -= qty
                    elif cur_pos > 0:
                        # Partially closing long; remainder opens short
                        short_open_qty = qty - cur_pos
                        net_positions[ticker] = -short_open_qty
                        short_queues[ticker].append((price, short_open_qty))
                    else:
                        # Position already 0 or negative — pure short entry
                        short_queues[ticker].append((price, qty))
                        net_positions[ticker] -= qty

                elif action == "buy":
                    if cur_pos < 0:
                        # Covering short position (FIFO)
                        cover_qty = min(qty, abs(cur_pos))
                        remaining = cover_qty
                        while remaining > 0 and short_queues[ticker]:
                            entry_price, entry_qty = short_queues[ticker][0]
                            closed = min(entry_qty, remaining)
                            pnl = (entry_price - price) * closed
                            short_pnls.append({
                                "ticker": ticker,
                                "timestamp": t["timestamp"],
                                "pnl": pnl,
                                "trade_type": "short",
                            })
                            remaining -= closed
                            if closed >= entry_qty:
                                short_queues[ticker].popleft()
                            else:
                                short_queues[ticker][0] = (entry_price, entry_qty - closed)
                    net_positions[ticker] += qty

            stats["short_count_evaluated"] = len(short_pnls)

            # --- Combine long + short PnLs for aggregate stats ---
            all_pnls = long_pnls + short_pnls
            if all_pnls:
                wins = sum(1 for p in all_pnls if p["pnl"] > 0)
                stats["win_rate"] = wins / len(all_pnls) * 100
                best = max(all_pnls, key=lambda x: x["pnl"])
                worst = min(all_pnls, key=lambda x: x["pnl"])
                stats["best_trade"] = best
                stats["worst_trade"] = worst

                wins_pnl = [p["pnl"] for p in all_pnls if p["pnl"] > 0]
                losses_pnl = [p["pnl"] for p in all_pnls if p["pnl"] <= 0]
                gross_wins = sum(wins_pnl)
                gross_losses = sum(losses_pnl)
                if gross_losses != 0:
                    stats["profit_factor"] = gross_wins / abs(gross_losses)
                elif gross_wins > 0:
                    stats["profit_factor"] = float("inf")
                stats["avg_win"] = sum(wins_pnl) / len(wins_pnl) if wins_pnl else None
                stats["avg_loss"] = sum(losses_pnl) / len(losses_pnl) if losses_pnl else None
    except Exception:
        pass
    return stats


_RF_ANNUAL = 0.043   # ~10Y Treasury approximation
_RF_DAILY  = _RF_ANNUAL / 252
_MIN_DAYS_FOR_RISK_METRICS = 20  # minimum trading days before ratios are meaningful


@st.cache_data(ttl=300)
def fetch_risk_metrics() -> dict:
    """Sharpe, Sortino, Max Drawdown, Calmar — resampled to daily from 15-min snapshots."""
    result = {"sharpe": None, "sortino": None, "max_drawdown_pct": None, "calmar": None,
              "n_days": 0, "insufficient_data": False}
    try:
        history = fetch_portfolio_history()
        if history.empty or len(history) < 2:
            return result
        daily = (
            history.set_index("timestamp")["total_value"]
            .resample("D")
            .last()
            .dropna()
        )
        if len(daily) < 2:
            return result
        daily_returns = daily.pct_change().dropna()
        result["n_days"] = len(daily_returns)
        if len(daily_returns) < 2:
            return result
        if len(daily_returns) < _MIN_DAYS_FOR_RISK_METRICS:
            result["insufficient_data"] = True
        mean_ret = daily_returns.mean()
        std_ret  = daily_returns.std(ddof=1)
        if std_ret and not np.isnan(std_ret) and std_ret > 0:
            result["sharpe"] = (mean_ret - _RF_DAILY) / std_ret * math.sqrt(252)
        downside = daily_returns[daily_returns < _RF_DAILY]
        if len(downside) == 0:
            result["sortino"] = float("inf")
        elif len(downside) >= 2:
            ds_std = downside.std(ddof=1)
            if ds_std > 0:
                result["sortino"] = (mean_ret - _RF_DAILY) / ds_std * math.sqrt(252)
        cum_max = daily.cummax()
        drawdowns = (daily - cum_max) / cum_max
        result["max_drawdown_pct"] = drawdowns.min() * 100
        n_days = len(daily)
        total_ret = (daily.iloc[-1] / daily.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1
        abs_max_dd = abs(drawdowns.min())
        if abs_max_dd > 0:
            result["calmar"] = ann_ret / abs_max_dd
        elif ann_ret >= 0:
            result["calmar"] = float("inf")
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)
def fetch_spy_comparison() -> dict:
    """SPY buy-and-hold vs portfolio, both normalized to INITIAL_CAPITAL. TTL=1hr."""
    result = {"portfolio_norm": None, "spy_norm": None, "spy_total_return_pct": None, "error": None}
    try:
        history = fetch_portfolio_history()
        if history.empty:
            result["error"] = "No portfolio history available."
            return result
        port_daily = (
            history.set_index("timestamp")["total_value"]
            .resample("D")
            .last()
            .dropna()
        )
        if len(port_daily) < 2:
            result["error"] = "Not enough portfolio history for comparison."
            return result
        start_date = port_daily.index[0].date()
        end_date   = port_daily.index[-1].date()
        spy_raw = yf.download(
            "SPY",
            start=str(start_date),
            end=str(end_date + pd.Timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
        if spy_raw.empty:
            result["error"] = "SPY data unavailable."
            return result
        spy_close = spy_raw["Close"].squeeze()
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
        port_daily.index = port_daily.index.tz_localize(None) if port_daily.index.tzinfo else port_daily.index
        port_aligned = port_daily.reindex(spy_close.index, method="ffill")
        first_spy  = spy_close.iloc[0]
        first_port = port_aligned.iloc[0]
        if first_spy == 0 or first_port == 0 or pd.isna(first_port):
            result["error"] = "Could not normalize — missing data at start date."
            return result
        result["spy_norm"]              = spy_close / first_spy * INITIAL_CAPITAL
        result["portfolio_norm"]        = port_aligned / first_port * INITIAL_CAPITAL
        result["spy_total_return_pct"]  = (spy_close.iloc[-1] / first_spy - 1) * 100
    except Exception:
        result["error"] = "SPY data unavailable."
    return result


# --- Sidebar ---
st.sidebar.title("Trading Agent Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Portfolio Overview",
        "Positions",
        "Trade History",
        "Fundamentals & Macro",
        "Agent Activity",
        "Performance Analytics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Auto-Refresh")
refresh_interval = st.sidebar.number_input(
    "Refresh interval (seconds)", min_value=10, max_value=300, value=60, step=10
)
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)

if auto_refresh:
    placeholder = st.sidebar.empty()
    placeholder.info(f"Refreshing every {refresh_interval}s")
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"DB: {DB_PATH}")


# --- Page: Portfolio Overview ---
if page == "Portfolio Overview":
    st.title("Portfolio Overview")

    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }
    </style>
    """, unsafe_allow_html=True)

    snapshot = fetch_latest_snapshot()
    _live_prices_ov = fetch_live_prices()
    _positions_ov = fetch_positions()
    _day_open = fetch_day_open_snapshot()

    col1, col2, col3, col4, col5 = st.columns(5)

    # Alpaca is the source of truth for portfolio value.
    # The local DB positions table only tracks long positions; Alpaca includes
    # short positions too.  We use Alpaca equity/positions directly, and fall
    # back to the DB snapshot only if the API is unreachable.
    _alpaca = fetch_alpaca_account()

    if _alpaca:
        total_value    = _alpaca["equity"]
        cash           = _alpaca["cash"]
        buying_power   = _alpaca["buying_power"]
        long_mv        = _alpaca["long_market_value"]
        short_mv       = _alpaca["short_market_value"]
        unrealized_pnl = _alpaca["unrealized_pnl"]
        realized_pnl   = total_value - INITIAL_CAPITAL - unrealized_pnl
        last_equity    = _alpaca.get("last_equity")
        if last_equity:
            daily_pnl = total_value - last_equity
        elif _day_open:
            daily_pnl = total_value - (_day_open.get("total_value") or total_value)
        else:
            daily_pnl = snapshot.get("daily_pnl") or 0.0 if snapshot else 0.0
        n_long  = _alpaca.get("n_long", 0)
        n_short = _alpaca.get("n_short", 0)
        source_label = f"Alpaca live  |  {n_long} long, {n_short} short"
    elif snapshot:
        cash           = snapshot.get("cash") or 0.0
        total_value    = snapshot.get("total_value") or 0.0
        unrealized_pnl = snapshot.get("unrealized_pnl") or 0.0
        realized_pnl   = snapshot.get("realized_pnl") or 0.0
        daily_pnl      = snapshot.get("daily_pnl") or 0.0
        buying_power   = 0.0
        long_mv        = snapshot.get("positions_value") or 0.0
        short_mv       = 0.0
        source_label   = "DB snapshot (Alpaca unavailable — agent may not be running)"
        n_long = n_short = 0
    else:
        for col in [col1, col2, col3, col4, col5]:
            col.metric("—", "No data")
        source_label = ""
        total_value = unrealized_pnl = realized_pnl = daily_pnl = cash = buying_power = long_mv = short_mv = 0.0
        n_long = n_short = 0

    if total_value:
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric(
            "Unrealized P&L",
            f"${unrealized_pnl:,.2f}",
            delta=f"{unrealized_pnl / total_value * 100:+.2f}%" if total_value else None,
        )
        col3.metric(
            "Today's P&L",
            f"${daily_pnl:,.2f}",
            delta=f"{daily_pnl / (total_value - daily_pnl) * 100:+.2f}%" if (total_value - daily_pnl) else None,
        )
        col4.metric(
            "Realized P&L",
            f"${realized_pnl:,.2f}",
            delta=f"{realized_pnl / INITIAL_CAPITAL * 100:+.2f}%" if INITIAL_CAPITAL else None,
        )
        col5.metric("Buying Power", f"${buying_power:,.2f}")

        # Exposure breakdown row
        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
        exp_c1.metric("Long Exposure", f"${long_mv:,.2f}")
        exp_c2.metric("Short Exposure", f"${short_mv:,.2f}")
        exp_c3.metric("Cash", f"${cash:,.2f}")
        if cash < 0:
            exp_c3.caption(f"Margin borrowed: ${abs(cash):,.2f}")
        leverage = (long_mv + abs(short_mv)) / total_value if total_value else 0
        exp_c4.metric("Gross Leverage", f"{leverage:.2f}x")

    if not _live_prices_ov.empty:
        freshest = _live_prices_ov["price_updated_at"].max()
        st.caption(f"Prices: Finnhub WebSocket  |  Last updated: {_to_et(freshest)}  |  Source: {source_label}")
    else:
        st.caption(f"Source: {source_label}")

    st.markdown("---")
    st.subheader("Portfolio Value Over Time")

    history = fetch_portfolio_history()
    if history.empty:
        st.info("No portfolio history yet.")
    else:
        # Correction boundary: snapshots before this timestamp used long-only accounting
        # and overstate portfolio value by ~$57k (short positions were ignored).
        _correction_ts = pd.Timestamp("2026-03-01T19:10:41", tz="UTC").tz_convert("America/New_York")
        pre = history[history["timestamp"] < _correction_ts]
        post = history[history["timestamp"] >= _correction_ts]

        fig = go.Figure()
        if not pre.empty:
            fig.add_trace(go.Scatter(
                x=pre["timestamp"], y=pre["total_value"],
                mode="lines", name="Portfolio Value (long-only, overstated)",
                line=dict(color="#aec7e8", width=2, dash="dot"),
            ))
        if not post.empty:
            fig.add_trace(go.Scatter(
                x=post["timestamp"], y=post["total_value"],
                mode="lines", name="Portfolio Value (Alpaca equity, accurate)",
                line=dict(color="#1f77b4", width=2),
            ))
        fig.add_hline(
            y=INITIAL_CAPITAL,
            line_dash="dash", line_color="gray",
            annotation_text=f"Starting Capital ${INITIAL_CAPITAL:,.0f}",
            annotation_position="bottom right",
        )
        if not pre.empty and not post.empty:
            fig.add_shape(
                type="line",
                x0=_correction_ts.isoformat(), x1=_correction_ts.isoformat(),
                y0=0, y1=1, yref="paper",
                line=dict(color="orange", dash="dash", width=1.5),
            )
            fig.add_annotation(
                x=_correction_ts.isoformat(), y=1, yref="paper",
                text="Short tracking added",
                showarrow=False, xanchor="right",
                font=dict(color="orange", size=11),
            )
        fig.update_layout(
            xaxis_title="Time", yaxis_title="Value ($)",
            hovermode="x unified", height=400,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        if not pre.empty:
            st.caption(
                "Dotted line = snapshots recorded before short-position tracking was added (2026-03-01). "
                "Those values reflect long positions + cash only and overstate true portfolio value by ~$57k. "
                "Solid line = Alpaca equity (accurate, includes shorts)."
            )

    st.markdown("---")
    st.subheader("Estimated Tax Liability")

    if not total_value:
        st.info("No snapshot data available.")
    else:
        taxable_realized = max(0.0, realized_pnl)
        taxable_unrealized = max(0.0, unrealized_pnl)

        st.caption(
            "Federal short-term rates only (active trading is typically held < 1 year). "
            "Does not include state taxes, the 3.8% NIIT, or other income. "
            "Consult a tax professional for actual liability."
        )

        brackets = [
            ("10%", 0.10),
            ("12%", 0.12),
            ("22%", 0.22),
            ("24%", 0.24),
            ("32%", 0.32),
            ("35%", 0.35),
            ("37%", 0.37),
        ]

        rows = []
        for rate_label, rate in brackets:
            accrued = taxable_realized * rate
            deferred = taxable_unrealized * rate
            total_exposure = accrued + deferred
            after_tax_value = total_value - accrued
            rows.append({
                "Bracket": rate_label,
                "Accrued (Realized)": f"${accrued:,.2f}",
                "Deferred (Unrealized)": f"${deferred:,.2f}",
                "Total Exposure": f"${total_exposure:,.2f}",
                "After-Tax Portfolio": f"${after_tax_value:,.2f}",
            })

        st.table(pd.DataFrame(rows))
        st.caption(
            f"**Accrued** = tax owed now on closed trades (realized P&L: **${realized_pnl:,.2f}**)  |  "
            f"**Deferred** = additional tax if you sold all open positions today "
            f"(unrealized P&L: **${unrealized_pnl:,.2f}**)  |  "
            f"**After-Tax Portfolio** = current total value minus accrued tax only"
        )


# --- Page: Positions ---
elif page == "Positions":
    st.title("Current Positions")

    _alpaca_pos = fetch_alpaca_account()
    live_prices_df = fetch_live_prices()

    if _alpaca_pos:
        # Pull live positions directly from Alpaca (long + short)
        try:
            import os as _os
            from alpaca.trading.client import TradingClient as _TC2
            _paper2 = _os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
            _client2 = _TC2(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=_paper2)
            _alpaca_positions = _client2.get_all_positions()
        except Exception:
            _alpaca_positions = []

        if not _alpaca_positions:
            st.info("No open positions.")
        else:
            rows = []
            for p in _alpaca_positions:
                qty = float(p.qty)
                side = "Short" if qty < 0 else "Long"
                avg_entry = float(p.avg_entry_price)
                current_price = float(p.current_price)
                market_value = float(p.market_value)
                unrealized_pl = float(p.unrealized_pl)
                unrealized_plpc = float(p.unrealized_plpc) * 100
                rows.append({
                    "Ticker": p.symbol,
                    "Side": side,
                    "Quantity": qty,
                    "Avg Entry": avg_entry,
                    "Current Price": current_price,
                    "Market Value": market_value,
                    "Unrealized P&L": unrealized_pl,
                    "Unrealized P&L %": unrealized_plpc,
                })

            df = pd.DataFrame(rows).sort_values(["Side", "Ticker"])

            # Totals row
            total_market_value = df["Market Value"].sum()
            total_unrealized_pnl = df["Unrealized P&L"].sum()
            total_cost = df.apply(lambda r: abs(r["Quantity"]) * r["Avg Entry"], axis=1).sum()
            total_unrealized_pct = (total_unrealized_pnl / total_cost * 100) if total_cost else 0.0
            total_row = pd.DataFrame([{
                "Ticker": "TOTAL", "Side": "",
                "Quantity": float("nan"), "Avg Entry": float("nan"),
                "Current Price": float("nan"),
                "Market Value": total_market_value,
                "Unrealized P&L": total_unrealized_pnl,
                "Unrealized P&L %": total_unrealized_pct,
            }])
            display_df = pd.concat([df, total_row], ignore_index=True)

            format_map = {
                "Avg Entry": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Market Value": "${:,.2f}",
                "Unrealized P&L": "${:,.2f}",
                "Unrealized P&L %": "{:.2f}%",
            }
            styled = display_df.style.format(format_map, na_rep="N/A")
            st.dataframe(styled, use_container_width=True)
            st.caption("Live data from Alpaca — includes both long and short positions.")

            st.markdown("---")
            st.subheader("Position Allocation")

            # Pie chart: use absolute market value for sizing; label shorts distinctly
            pie_data = df.copy()
            pie_data["abs_market_value"] = pie_data["Market Value"].abs()
            pie_data["label"] = pie_data.apply(
                lambda r: f"{r['Ticker']} ({'S' if r['Side'] == 'Short' else 'L'})", axis=1
            )
            cash_value = max(_alpaca_pos.get("cash", 0.0), 0.0)
            cash_row = pd.DataFrame([{"label": "Cash", "abs_market_value": cash_value}])
            pie_data = pd.concat([pie_data[["label", "abs_market_value"]], cash_row], ignore_index=True)

            fig = go.Figure(go.Pie(
                labels=pie_data["label"].tolist(),
                values=pie_data["abs_market_value"].tolist(),
                hole=0.3, textinfo="percent", textposition="inside",
                textfont=dict(size=13),
            ))
            fig.update_layout(
                title="Portfolio Allocation (abs market value; S=Short, L=Long)",
                height=450, margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to DB positions (long-only) when Alpaca is unreachable
        st.warning("Alpaca unavailable — showing long positions from DB only (short positions not reflected).")
        positions_df = fetch_positions()
        live_prices_df = fetch_live_prices()
        snapshot = fetch_latest_snapshot()

        if positions_df.empty:
            st.info("No open positions.")
        else:
            df = positions_df.copy()
            if not live_prices_df.empty:
                df = df.merge(live_prices_df[["ticker", "live_price", "price_updated_at"]],
                              on="ticker", how="left")
            else:
                df["live_price"] = None
            df["effective_price"] = df["live_price"].fillna(df["current_price"]).fillna(df["avg_cost"])
            df["market_value"] = df["effective_price"] * df["quantity"]
            df["unrealized_pnl"] = (df["effective_price"] - df["avg_cost"]) * df["quantity"]
            df["unrealized_pnl_pct"] = (df["effective_price"] - df["avg_cost"]) / df["avg_cost"] * 100
            st.dataframe(df[["ticker", "quantity", "avg_cost", "effective_price",
                              "market_value", "unrealized_pnl", "unrealized_pnl_pct"]],
                         use_container_width=True)


# --- Page: Trade History ---
elif page == "Trade History":
    st.title("Trade History")

    tickers = ["All"] + fetch_all_tickers_traded()
    selected_ticker = st.selectbox("Filter by ticker", tickers)

    trades_df = fetch_trades(ticker_filter=selected_ticker)

    if trades_df.empty:
        st.info("No trades recorded yet.")
    else:
        for _, row in trades_df.iterrows():
            action = str(row.get("action", "")).lower()
            color = "green" if action == "buy" else "red"
            action_label = action.upper()

            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
            col1.markdown(f"**{row.get('ticker', '')}**")
            col2.markdown(
                f"<span style='color:{color}'>**{action_label}**</span>",
                unsafe_allow_html=True,
            )
            col3.markdown(f"Qty: {row.get('quantity', 0):.2f}")
            col4.markdown(f"${row.get('price', 0):,.2f}")
            col5.markdown(_to_et(row.get("timestamp", "")))

            reasoning = row.get("reasoning") or ""
            confidence = row.get("confidence")
            if reasoning:
                with st.expander("Reasoning"):
                    if confidence is not None:
                        st.caption(f"Confidence: {confidence:.2f}")
                    st.write(reasoning)
            st.divider()


# --- Page: Fundamentals & Macro ---
elif page == "Fundamentals & Macro":
    st.title("Fundamentals & Macro")

    tab1, tab2 = st.tabs(["Company Fundamentals", "Macro Dashboard"])

    with tab1:
        st.subheader("Company Fundamentals")
        fund_df = fetch_fundamentals()
        if fund_df.empty:
            st.info("No fundamentals data yet.")
        else:
            format_map = {}
            for col in ["pe_ttm", "forward_pe", "gross_margin", "operating_margin", "roe", "dividend_yield"]:
                if col in fund_df.columns:
                    format_map[col] = "{:.2f}"
            if "avg_price_target" in fund_df.columns:
                format_map["avg_price_target"] = "${:,.2f}"

            styled = fund_df.style.format(format_map, na_rep="N/A")
            st.dataframe(styled, use_container_width=True)

    with tab2:
        st.subheader("Macro Dashboard")
        macro = fetch_latest_macro()
        if not macro:
            st.info("No macro data yet.")
        else:
            st.caption(f"Last updated: {_to_et(macro.get('timestamp', ''))}")

            col1, col2, col3 = st.columns(3)
            fed_rate = macro.get("fed_funds_rate")
            t10 = macro.get("treasury_10y")
            t2 = macro.get("treasury_2y")
            spread = macro.get("yield_curve_spread")
            cpi = macro.get("cpi_yoy")
            unemp = macro.get("unemployment_rate")
            gdp = macro.get("gdp_growth_qoq")
            vix = macro.get("vix")
            sentiment = macro.get("consumer_sentiment")
            jobless = macro.get("initial_jobless_claims")

            def fmt_pct(val):
                return f"{val:.2f}%" if val is not None else "N/A"

            def fmt_num(val, decimals=2):
                return f"{val:,.{decimals}f}" if val is not None else "N/A"

            col1.metric("Fed Funds Rate", fmt_pct(fed_rate))
            col1.metric("10Y Treasury", fmt_pct(t10))
            col1.metric("2Y Treasury", fmt_pct(t2))
            col1.metric("Yield Curve Spread", fmt_pct(spread))

            col2.metric("CPI YoY", fmt_pct(cpi))
            col2.metric("Unemployment Rate", fmt_pct(unemp))
            col2.metric("GDP Growth QoQ", fmt_pct(gdp))

            col3.metric("VIX", fmt_num(vix))
            col3.metric("Consumer Sentiment", fmt_num(sentiment))
            col3.metric("Initial Jobless Claims", fmt_num(jobless, 0))

            st.markdown("---")
            st.subheader("Derived Signals")

            signals = []
            if spread is not None:
                if spread < 0:
                    signals.append(("Yield Curve", "INVERTED (Recession Warning)", "red"))
                else:
                    signals.append(("Yield Curve", f"Normal (spread: {spread:.2f}%)", "green"))

            if vix is not None:
                if vix > 30:
                    signals.append(("VIX", f"VERY ELEVATED ({vix:.1f}) - High Fear", "red"))
                elif vix > 25:
                    signals.append(("VIX", f"Elevated ({vix:.1f}) - Caution", "orange"))
                else:
                    signals.append(("VIX", f"Normal ({vix:.1f})", "green"))

            if cpi is not None:
                if cpi > 4.0:
                    signals.append(("Inflation", f"HIGH ({cpi:.2f}% YoY) - Fed hawkish pressure", "red"))
                elif cpi > 2.5:
                    signals.append(("Inflation", f"Moderately Elevated ({cpi:.2f}% YoY)", "orange"))
                else:
                    signals.append(("Inflation", f"Near Target ({cpi:.2f}% YoY)", "green"))

            if unemp is not None:
                if unemp > 5.0:
                    signals.append(("Unemployment", f"ELEVATED ({unemp:.1f}%)", "red"))
                else:
                    signals.append(("Unemployment", f"Healthy ({unemp:.1f}%)", "green"))

            for label, text, color in signals:
                st.markdown(
                    f"**{label}:** <span style='color:{color}'>{text}</span>",
                    unsafe_allow_html=True,
                )


# --- Page: Agent Activity ---
elif page == "Agent Activity":
    st.title("Agent Activity")

    logs_df = fetch_agent_logs(limit=100)

    if logs_df.empty:
        st.info("No agent logs yet.")
    else:
        all_event_types = ["All"] + sorted(logs_df["event_type"].dropna().unique().tolist())
        selected_event = st.selectbox("Filter by event type", all_event_types)

        if selected_event != "All":
            filtered_df = logs_df[logs_df["event_type"] == selected_event].copy()
        else:
            filtered_df = logs_df.copy()

        st.subheader("Event Type Distribution")
        event_counts = logs_df["event_type"].value_counts().reset_index()
        event_counts.columns = ["event_type", "count"]
        fig = px.bar(
            event_counts,
            x="event_type",
            y="count",
            title="Log Count by Event Type",
            color="event_type",
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Event Type",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader(f"Recent Logs ({len(filtered_df)} shown)")

        for _, row in filtered_df.iterrows():
            ts = _to_et(row.get("timestamp", ""))
            event_type = row.get("event_type", "")
            ticker = row.get("related_ticker") or ""
            content_raw = row.get("content") or ""

            header = f"**{ts}** | `{event_type}`"
            if ticker:
                header += f" | {ticker}"

            with st.expander(header):
                try:
                    parsed = json.loads(content_raw)
                    st.json(parsed)
                except (json.JSONDecodeError, TypeError):
                    st.text(content_raw)


# --- Page: Performance Analytics ---
elif page == "Performance Analytics":
    st.title("Performance Analytics")

    stats = fetch_performance_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", stats["total_trades"])
    col2.metric("Buy Orders", stats["buy_count"])
    col3.metric("Sell Orders", stats["sell_count"])

    st.markdown("---")

    win_rate = stats.get("win_rate")
    n_long_eval = stats.get("sell_count_evaluated", 0)
    n_short_eval = stats.get("short_count_evaluated", 0)
    n_eval = n_long_eval + n_short_eval
    n_sells = stats.get("sell_count", 0)
    if win_rate is not None:
        st.metric(
            f"Win Rate ({n_long_eval}L+{n_short_eval}S/{n_sells} closed trades)",
            f"{win_rate:.2f}%",
        )
    else:
        st.metric("Win Rate", "N/A")

    st.markdown("---")
    st.subheader("Daily Returns")

    history = fetch_portfolio_history()
    if history.empty or "total_value" not in history.columns:
        st.info("No portfolio history for daily returns.")
    else:
        # Recompute daily P&L from day-over-day closing total_value.
        # This cancels the systematic long-only inflation for historical days
        # (both start and end of each day had the same overstatement, so the
        # delta is approximately correct). The correction boundary date (Mar 1)
        # is excluded so it doesn't mix inflated opens with real closes.
        _correction_date = "2026-03-01"
        _pre_hist = history[history["timestamp"].dt.date.astype(str) < _correction_date].copy()
        _post_hist = history[history["timestamp"].dt.date.astype(str) >= _correction_date].copy()

        daily_rows = []
        # Pre-correction: use day-over-day closing values
        if not _pre_hist.empty:
            _pre_hist["date"] = _pre_hist["timestamp"].dt.date
            _day_close = _pre_hist.groupby("date")["total_value"].last().reset_index()
            _day_close["daily_pnl"] = _day_close["total_value"].diff()
            _day_close.loc[_day_close.index[0], "daily_pnl"] = (
                _day_close["total_value"].iloc[0] - INITIAL_CAPITAL
            )
            daily_rows.append(_day_close[["date", "daily_pnl"]])
        # Post-correction: same approach but data is already accurate
        if not _post_hist.empty:
            _post_hist["date"] = _post_hist["timestamp"].dt.date
            _day_close_post = _post_hist.groupby("date")["total_value"].last().reset_index()
            if not _pre_hist.empty:
                # Link to last pre-correction close
                _prev_close = _pre_hist.groupby("date")["total_value"].last().iloc[-1]
            else:
                _prev_close = INITIAL_CAPITAL
            _day_close_post["daily_pnl"] = _day_close_post["total_value"].diff()
            _day_close_post.loc[_day_close_post.index[0], "daily_pnl"] = (
                _day_close_post["total_value"].iloc[0] - INITIAL_CAPITAL
            )
            daily_rows.append(_day_close_post[["date", "daily_pnl"]])

        if not daily_rows:
            st.info("No daily P&L data recorded yet.")
        else:
            daily = pd.concat(daily_rows, ignore_index=True)
            daily["date"] = pd.to_datetime(daily["date"])
            daily["daily_return_pct"] = daily["daily_pnl"] / INITIAL_CAPITAL * 100

            fig = go.Figure()
            colors = ["green" if v >= 0 else "red" for v in daily["daily_pnl"]]
            fig.add_trace(go.Bar(
                x=daily["date"], y=daily["daily_pnl"],
                marker_color=colors, name="Daily P&L",
            ))
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Daily P&L ($)",
                height=350, margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Use Alpaca equity for Total Return — historical snapshots only tracked
            # long positions and overstate portfolio value by ~$57k (short positions ignored).
            _alpaca_pa = fetch_alpaca_account()
            if _alpaca_pa:
                total_return = (_alpaca_pa["equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                current_value = _alpaca_pa["equity"]
            else:
                total_return = (
                    (history["total_value"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                    if not history.empty else 0.0
                )
                current_value = history["total_value"].iloc[-1] if not history.empty else INITIAL_CAPITAL
            col_a, col_b = st.columns(2)
            col_a.metric("Total Return", f"{total_return:.2f}%")
            col_b.metric("Starting Capital", f"${INITIAL_CAPITAL:,.2f}")

    # --- Risk-Adjusted Performance ---
    st.markdown("---")
    st.subheader("Risk-Adjusted Performance")

    risk = fetch_risk_metrics()
    n_days = risk.get("n_days", 0)
    insufficient = risk.get("insufficient_data", False)

    st.warning(
        "⚠ Historical snapshots were recorded with **long-only portfolio valuation** — "
        "short positions (TSLA, NVDA, BA, HD, CRM, etc.) were not included, causing snapshots "
        "to overstate portfolio value by ~$57k. Sharpe, Sortino, Drawdown and the chart below "
        "are computed from these inflated values and are **not reliable**. "
        "They will become accurate once the agent runs and records new corrected snapshots."
    )
    if insufficient:
        st.warning(
            f"⚠ Only {n_days} trading days of history — risk ratios require "
            f"{_MIN_DAYS_FOR_RISK_METRICS}+ days to be statistically meaningful. "
            "Values shown are mathematically correct but unreliable at this sample size."
        )

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    sharpe = risk["sharpe"]
    col_r1.metric(f"Sharpe Ratio (Ann., n={n_days}d)", f"{sharpe:.2f}" if sharpe is not None else "N/A")

    sortino = risk["sortino"]
    if sortino is None:
        col_r2.metric("Sortino Ratio (Ann.)", "N/A")
    elif sortino == float("inf"):
        col_r2.metric("Sortino Ratio (Ann.)", "∞ (no down days)")
    else:
        col_r2.metric("Sortino Ratio (Ann.)", f"{sortino:.2f}")

    mdd = risk["max_drawdown_pct"]
    col_r3.metric("Max Drawdown (close-to-close)", f"{mdd:.2f}%" if mdd is not None else "N/A")

    calmar = risk["calmar"]
    if calmar is None:
        col_r4.metric("Calmar Ratio", "N/A")
    elif calmar == float("inf"):
        col_r4.metric("Calmar Ratio", "∞")
    else:
        col_r4.metric("Calmar Ratio", f"{calmar:.2f}")

    st.caption(
        "Sharpe & Sortino annualized from **close-to-close** daily returns (252 trading days/yr). "
        "Risk-free rate: 4.3% (10Y Treasury approx.). "
        "Max Drawdown measures close-to-close — intraday dips appear in the Daily Returns chart above. "
        "Calmar = Annualized Return / |Max Drawdown|."
    )

    # --- Trade Quality ---
    st.markdown("---")
    st.subheader("Trade Quality")

    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)

    pf = stats.get("profit_factor")
    if pf is None:
        col_q1.metric("Profit Factor", "N/A")
    elif pf == float("inf"):
        col_q1.metric("Profit Factor", "∞ (all winners)")
    else:
        col_q1.metric("Profit Factor", f"{pf:.2f}")

    avg_win = stats.get("avg_win")
    col_q2.metric("Avg Win / Trade", f"${avg_win:,.2f}" if avg_win is not None else "N/A")

    avg_loss = stats.get("avg_loss")
    col_q3.metric("Avg Loss / Trade", f"${avg_loss:,.2f}" if avg_loss is not None else "N/A")

    best = stats.get("best_trade")
    if best:
        col_q4.metric(
            f"Best Trade ({best['ticker']} @ {_to_et(best['timestamp'], date_only=True)})",
            f"${best['pnl']:,.2f}",
        )
    else:
        col_q4.metric("Best Trade", "N/A")

    worst = stats.get("worst_trade")
    if worst:
        col_q5.metric(
            f"Worst Trade ({worst['ticker']} @ {_to_et(worst['timestamp'], date_only=True)})",
            f"${worst['pnl']:,.2f}",
        )
    else:
        col_q5.metric("Worst Trade", "N/A")

    st.caption("Based on all closed trades (long closes + covered shorts). Profit Factor = Gross Wins / |Gross Losses|.")

    # --- SPY Comparison ---
    st.markdown("---")
    st.subheader("Portfolio vs. Buy-and-Hold SPY")

    spy_data = fetch_spy_comparison()

    if spy_data["error"]:
        st.warning(f"SPY comparison unavailable: {spy_data['error']}")
    else:
        # Use Alpaca equity for the return comparison (same fix as Total Return above)
        _alpaca_spy = fetch_alpaca_account()
        if _alpaca_spy:
            port_total_return = (_alpaca_spy["equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        else:
            port_total_return = (
                (history["total_value"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                if not history.empty else 0.0
            )
        spy_return = spy_data["spy_total_return_pct"]

        col_s1, col_s2 = st.columns(2)
        col_s1.metric(
            "Portfolio Total Return",
            f"{port_total_return:.2f}%",
            delta=f"{port_total_return - spy_return:+.2f}% vs SPY",
        )
        col_s2.metric("SPY Total Return (same period)", f"{spy_return:.2f}%")

        port_norm = spy_data["portfolio_norm"]
        spy_norm  = spy_data["spy_norm"]

        fig_spy = go.Figure()
        fig_spy.add_trace(go.Scatter(
            x=port_norm.index,
            y=port_norm.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_spy.add_trace(go.Scatter(
            x=spy_norm.index,
            y=spy_norm.values,
            mode="lines",
            name="SPY Buy-and-Hold",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig_spy.add_hline(
            y=INITIAL_CAPITAL,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Starting Capital ${INITIAL_CAPITAL:,.0f}",
            annotation_position="bottom right",
        )
        fig_spy.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Value (normalized to ${INITIAL_CAPITAL:,.0f})",
            hovermode="x unified",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_spy, use_container_width=True)
        st.caption(
            "Both series normalized to starting capital. "
            "SPY adjusted close via yfinance (dividend/split-adjusted)."
        )
