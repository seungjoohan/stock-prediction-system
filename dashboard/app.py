import json
import os
import sqlite3
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DB_PATH, INITIAL_CAPITAL

st.set_page_config(page_title="Trading Agent Dashboard", layout="wide")


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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
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

            sell_rows = conn.execute(
                "SELECT t.ticker, t.price as sell_price, t.quantity, t.total_value, t.timestamp, "
                "p.avg_cost "
                "FROM trades t "
                "LEFT JOIN positions p ON t.ticker = p.ticker "
                "WHERE t.action = 'sell'",
            ).fetchall()

            if sell_rows:
                pnls = []
                for r in sell_rows:
                    if r["avg_cost"] is not None:
                        pnl = (r["sell_price"] - r["avg_cost"]) * r["quantity"]
                        pnls.append(
                            {
                                "ticker": r["ticker"],
                                "timestamp": r["timestamp"],
                                "pnl": pnl,
                                "sell_price": r["sell_price"],
                            }
                        )
                if pnls:
                    wins = sum(1 for p in pnls if p["pnl"] > 0)
                    stats["win_rate"] = wins / len(pnls) * 100
                    best = max(pnls, key=lambda x: x["pnl"])
                    worst = min(pnls, key=lambda x: x["pnl"])
                    stats["best_trade"] = best
                    stats["worst_trade"] = worst
    except Exception:
        pass
    return stats


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

    col1, col2, col3, col4, col5 = st.columns(5)
    if snapshot:
        total_value = snapshot.get("total_value") or 0.0
        positions_value = snapshot.get("positions_value") or 0.0
        cash = snapshot.get("cash") or 0.0
        daily_pnl = snapshot.get("daily_pnl") or 0.0
        realized_pnl = snapshot.get("realized_pnl") or 0.0
        unrealized_pnl = snapshot.get("unrealized_pnl") or 0.0

        cost_basis = positions_value - unrealized_pnl

        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric(
            "Asset Value",
            f"${positions_value:,.2f}",
            delta=f"+${unrealized_pnl:,.2f}" if unrealized_pnl >= 0 else f"-${abs(unrealized_pnl):,.2f}",
        )
        col2.caption(f"Cost basis: ${cost_basis:,.2f}")
        col3.metric("Cash", f"${cash:,.2f}")
        col4.metric(
            "Today's P&L",
            f"${daily_pnl:,.2f}",
            delta=f"{daily_pnl / total_value * 100:+.2f}%" if total_value else None,
        )
        col5.metric(
            "Realized P&L",
            f"${realized_pnl:,.2f}",
            delta=f"{realized_pnl / total_value * 100:+.2f}%" if total_value else None,
        )
    else:
        for col in [col1, col2, col3, col4, col5]:
            col.metric("—", "No data")

    st.markdown("---")
    st.subheader("Portfolio Value Over Time")

    history = fetch_portfolio_history()
    if history.empty:
        st.info("No portfolio history yet.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_hline(
            y=INITIAL_CAPITAL,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Starting Capital ${INITIAL_CAPITAL:,.0f}",
            annotation_position="bottom right",
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value ($)",
            hovermode="x unified",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Page: Positions ---
elif page == "Positions":
    st.title("Current Positions")

    positions_df = fetch_positions()
    live_prices_df = fetch_live_prices()
    snapshot = fetch_latest_snapshot()

    if positions_df.empty:
        st.info("No open positions.")
    else:
        df = positions_df.copy()

        # Merge live WebSocket prices; fall back to Alpaca-synced current_price
        if not live_prices_df.empty:
            df = df.merge(live_prices_df[["ticker", "live_price", "price_updated_at"]],
                          on="ticker", how="left")
        else:
            df["live_price"] = None
            df["price_updated_at"] = None

        # Coalesce: live WebSocket price → Alpaca-synced price → avg_cost
        df["effective_price"] = (
            df["live_price"]
            .fillna(df["current_price"])
            .fillna(df["avg_cost"])
        )

        df["market_value"] = df["effective_price"] * df["quantity"]
        df["unrealized_pnl"] = (df["effective_price"] - df["avg_cost"]) * df["quantity"]
        df["unrealized_pnl_pct"] = (
            (df["effective_price"] - df["avg_cost"]) / df["avg_cost"] * 100
        )

        display_df = df[
            [
                "ticker",
                "quantity",
                "avg_cost",
                "effective_price",
                "market_value",
                "unrealized_pnl",
                "unrealized_pnl_pct",
                "updated_at",
            ]
        ].copy()

        display_df = display_df.rename(
            columns={
                "ticker": "Ticker",
                "quantity": "Quantity",
                "avg_cost": "Avg Cost",
                "effective_price": "Current Price",
                "market_value": "Market Value",
                "unrealized_pnl": "Unrealized P&L",
                "unrealized_pnl_pct": "Unrealized P&L %",
                "updated_at": "Last Synced",
            }
        )

        # Totals row
        total_cost_basis = (df["avg_cost"] * df["quantity"]).sum()
        total_market_value = df["market_value"].sum()
        total_unrealized_pnl = df["unrealized_pnl"].sum()
        total_unrealized_pnl_pct = (
            (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis else 0.0
        )
        total_row = pd.DataFrame([{
            "Ticker": "TOTAL",
            "Quantity": float("nan"),
            "Avg Cost": float("nan"),
            "Current Price": float("nan"),
            "Market Value": total_market_value,
            "Unrealized P&L": total_unrealized_pnl,
            "Unrealized P&L %": total_unrealized_pnl_pct,
            "Last Synced": "",
        }])
        display_df = pd.concat([display_df, total_row], ignore_index=True)

        format_map = {
            "Avg Cost": "${:,.2f}",
            "Current Price": "${:,.2f}",
            "Market Value": "${:,.2f}",
            "Unrealized P&L": "${:,.2f}",
            "Unrealized P&L %": "{:.2f}%",
        }
        styled = display_df.style.format(format_map, na_rep="N/A")
        st.dataframe(styled, use_container_width=True)

        # Price source indicator
        if not live_prices_df.empty and df["live_price"].notna().any():
            freshest = df["price_updated_at"].dropna().max()
            st.caption(f"Prices: Finnhub WebSocket  |  Last updated: {str(freshest)[:19]} UTC")
        else:
            st.caption("Prices: Alpaca sync (live prices not yet available — agent may not be running)")

        st.markdown("---")
        st.subheader("Position Allocation")

        pie_data = df[["ticker", "market_value"]].copy()
        pie_data["market_value"] = pd.to_numeric(pie_data["market_value"], errors="coerce").fillna(0.0)

        # Derive cash from total_value - positions so negative Alpaca margin cash doesn't break the chart
        total_value = snapshot.get("total_value", 0.0) if snapshot else 0.0
        cash_value = max(total_value - pie_data["market_value"].sum(), 0.0)
        cash_row = pd.DataFrame([{"ticker": "Cash", "market_value": cash_value}])
        pie_data = pd.concat([pie_data, cash_row], ignore_index=True)

        fig = go.Figure(go.Pie(
            labels=pie_data["ticker"].tolist(),
            values=pie_data["market_value"].tolist(),
            hole=0.3,
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=13),
        ))
        fig.update_layout(
            title="Portfolio Allocation",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"),
        )
        st.plotly_chart(fig, use_container_width=True)


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
            col5.markdown(str(row.get("timestamp", ""))[:19])

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
            st.caption(f"Last updated: {str(macro.get('timestamp', ''))[:19]}")

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
            ts = str(row.get("timestamp", ""))[:19]
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

    col4, col5 = st.columns(2)
    win_rate = stats.get("win_rate")
    if win_rate is not None:
        col4.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        col4.metric("Win Rate", "N/A")

    best = stats.get("best_trade")
    worst = stats.get("worst_trade")

    if best:
        col5.metric(
            f"Best Trade ({best['ticker']} @ {str(best['timestamp'])[:10]})",
            f"${best['pnl']:,.2f}",
        )

    if worst:
        st.metric(
            f"Worst Trade ({worst['ticker']} @ {str(worst['timestamp'])[:10]})",
            f"${worst['pnl']:,.2f}",
        )

    st.markdown("---")
    st.subheader("Daily Returns")

    history = fetch_portfolio_history()
    if history.empty or "daily_pnl" not in history.columns:
        st.info("No portfolio history for daily returns.")
    else:
        daily = history.dropna(subset=["daily_pnl"]).copy()
        if daily.empty:
            st.info("No daily P&L data recorded yet.")
        else:
            daily["daily_return_pct"] = daily["daily_pnl"] / INITIAL_CAPITAL * 100

            fig = go.Figure()
            colors = ["green" if v >= 0 else "red" for v in daily["daily_pnl"]]
            fig.add_trace(
                go.Bar(
                    x=daily["timestamp"],
                    y=daily["daily_pnl"],
                    marker_color=colors,
                    name="Daily P&L",
                )
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Daily P&L ($)",
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            total_return = (
                (history["total_value"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                if not history.empty
                else 0.0
            )
            col_a, col_b = st.columns(2)
            col_a.metric("Total Return", f"{total_return:.2f}%")
            col_b.metric(
                "Starting Capital",
                f"${INITIAL_CAPITAL:,.2f}",
            )
