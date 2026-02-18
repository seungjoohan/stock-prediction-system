import json
import logging
import re
import time

from agent.prompts import (
    DECISION_SYSTEM_PROMPT,
    DECISION_USER_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_PROMPT,
)
from agent.risk_manager import TradeAction
from db.database import insert_agent_log
from services.llm_provider import LLMProvider
from services.news_ingestion import NewsItem

logger = logging.getLogger(__name__)

_BATCH_SIZE = 5


def _extract_json(text: str) -> str:
    """Strip markdown code fences and extract the first JSON array or object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("[")
    if start != -1:
        return text[start:]
    start = text.find("{")
    if start != -1:
        return text[start:]
    return text


def _parse_json_list(raw: str) -> list:
    """Parse a JSON list from raw LLM output; return empty list on failure."""
    try:
        candidate = _extract_json(raw)
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
        logger.warning("LLM returned JSON but not a list: %s", type(parsed))
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("JSON parse failure: %s — raw snippet: %.200s", exc, raw)
        return []


def _format_news_items(news_items: list[NewsItem]) -> str:
    lines = []
    for idx, item in enumerate(news_items, start=1):
        tickers_str = ", ".join(item.tickers) if item.tickers else "N/A"
        lines.append(
            f"{idx}. [{item.source}] {item.headline}\n"
            f"   Summary: {item.summary or 'N/A'}\n"
            f"   Tickers: {tickers_str}"
        )
    return "\n\n".join(lines)


def _format_portfolio_summary(portfolio_state: dict) -> str:
    cash = portfolio_state.get("cash", 0.0)
    total_value = portfolio_state.get("total_value", 0.0)
    daily_pnl = portfolio_state.get("daily_pnl", 0.0)
    positions = portfolio_state.get("positions", [])

    lines = [
        f"Cash: ${cash:,.2f}",
        f"Total portfolio value: ${total_value:,.2f}",
        f"Daily P&L: ${daily_pnl:,.2f}",
        f"Number of open positions: {len(positions)}",
    ]

    if positions:
        lines.append("Positions:")
        for pos in positions:
            ticker = pos.get("ticker", "?")
            qty = pos.get("quantity", 0)
            price = pos.get("current_price", 0.0)
            avg_cost = pos.get("avg_cost", 0.0)
            pos_value = qty * price
            unrealized = (price - avg_cost) * qty if avg_cost else 0.0
            lines.append(
                f"  {ticker}: {qty} shares @ ${price:.2f} "
                f"(avg cost ${avg_cost:.2f}, value ${pos_value:,.2f}, "
                f"unrealized P&L ${unrealized:,.2f})"
            )

    return "\n".join(lines)


def _format_sentiment_signals(sentiment_signals: list[dict]) -> str:
    if not sentiment_signals:
        return "No sentiment signals available."
    lines = []
    for sig in sentiment_signals:
        ticker = sig.get("ticker", "?")
        sentiment = sig.get("sentiment", 0.0)
        confidence = sig.get("confidence", 0.0)
        horizon = sig.get("impact_horizon", "unknown")
        relevance = sig.get("relevance_score", 0.0)
        reasoning = sig.get("reasoning", "")
        lines.append(
            f"- {ticker}: sentiment={sentiment:+.2f}, confidence={confidence:.2f}, "
            f"horizon={horizon}, relevance={relevance:.2f} — {reasoning}"
        )
    return "\n".join(lines)


def _format_price_data(current_prices: dict[str, float]) -> str:
    if not current_prices:
        return "No price data available."
    lines = []
    for ticker, price in current_prices.items():
        if isinstance(price, dict):
            close = price.get("close", price.get("price", 0.0))
            change_pct = price.get("change_pct")
            volume = price.get("volume")
            rsi = price.get("rsi")
            parts = [f"{ticker}: ${close:.2f}"]
            if change_pct is not None:
                parts.append(f"change={change_pct:+.2f}%")
            if volume is not None:
                parts.append(f"volume={volume:,.0f}")
            if rsi is not None:
                parts.append(f"RSI={rsi:.1f}")
            lines.append("  ".join(parts))
        else:
            lines.append(f"{ticker}: ${float(price):.2f}")
    return "\n".join(lines)


def _format_fundamentals(fundamentals: dict[str, dict]) -> str:
    if not fundamentals:
        return "No fundamentals data available."
    lines = []
    for ticker, data in fundamentals.items():
        pe = data.get("pe_ttm")
        fwd_pe = data.get("forward_pe")
        gross_margin = data.get("gross_margin")
        operating_margin = data.get("operating_margin")
        analyst_buy = data.get("analyst_buy")
        analyst_hold = data.get("analyst_hold")
        analyst_sell = data.get("analyst_sell")
        avg_target = data.get("avg_price_target")
        earnings_surprise = data.get("earnings_surprise_pct")

        parts = [ticker + ":"]
        if pe is not None:
            parts.append(f"P/E(TTM)={pe:.1f}")
        if fwd_pe is not None:
            parts.append(f"Fwd P/E={fwd_pe:.1f}")
        if gross_margin is not None:
            parts.append(f"Gross margin={gross_margin:.1%}")
        if operating_margin is not None:
            parts.append(f"Op margin={operating_margin:.1%}")
        if analyst_buy is not None or analyst_hold is not None or analyst_sell is not None:
            parts.append(
                f"Analysts: {analyst_buy or 0}B/{analyst_hold or 0}H/{analyst_sell or 0}S"
            )
        if avg_target is not None:
            parts.append(f"Avg target=${avg_target:.2f}")
        if earnings_surprise is not None:
            parts.append(f"EPS surprise={earnings_surprise:+.1f}%")

        lines.append("  ".join(parts))
    return "\n".join(lines)


def _format_macro_summary(macro_snapshot: dict | None) -> str:
    if not macro_snapshot:
        return "No macro data available."

    fed_rate = macro_snapshot.get("fed_funds_rate")
    t10y = macro_snapshot.get("treasury_10y")
    t2y = macro_snapshot.get("treasury_2y")
    yield_spread = macro_snapshot.get("yield_curve_spread")
    cpi = macro_snapshot.get("cpi_yoy")
    unemployment = macro_snapshot.get("unemployment_rate")
    gdp = macro_snapshot.get("gdp_growth_qoq")
    vix = macro_snapshot.get("vix")
    consumer_sentiment = macro_snapshot.get("consumer_sentiment")

    lines = []
    if fed_rate is not None:
        lines.append(f"Fed funds rate: {fed_rate:.2f}%")
    if t10y is not None:
        lines.append(f"10Y Treasury: {t10y:.2f}%")
    if t2y is not None:
        lines.append(f"2Y Treasury: {t2y:.2f}%")
    if yield_spread is not None:
        curve_signal = "normal" if yield_spread > 0 else "inverted (recession signal)"
        lines.append(f"Yield curve spread (10Y-2Y): {yield_spread:.2f}% — {curve_signal}")
    if cpi is not None:
        inflation_signal = "elevated" if cpi > 3.0 else "moderate" if cpi > 2.0 else "below target"
        lines.append(f"CPI (YoY): {cpi:.1f}% — {inflation_signal}")
    if unemployment is not None:
        lines.append(f"Unemployment: {unemployment:.1f}%")
    if gdp is not None:
        lines.append(f"GDP growth (QoQ): {gdp:.2f}%")
    if vix is not None:
        risk_signal = "high fear / risk-off" if vix > 30 else "elevated" if vix > 20 else "low volatility / risk-on"
        lines.append(f"VIX: {vix:.1f} — {risk_signal}")
    if consumer_sentiment is not None:
        lines.append(f"Consumer sentiment: {consumer_sentiment:.1f}")

    if not lines:
        return "Macro data present but no recognized fields."
    return "\n".join(lines)


class DecisionEngine:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def analyze_sentiment(self, news_items: list[NewsItem]) -> list[dict]:
        """Batch news items (up to 5 per call) and analyze sentiment via LLM."""
        if not news_items:
            return []

        results: list[dict] = []

        for batch_idx, batch_start in enumerate(range(0, len(news_items), _BATCH_SIZE)):
            # Pace requests: 2s gap between batches keeps us ~30 RPM max
            if batch_idx > 0:
                time.sleep(2)

            batch = news_items[batch_start: batch_start + _BATCH_SIZE]
            news_text = _format_news_items(batch)
            prompt = SENTIMENT_USER_PROMPT.format(news_items=news_text)

            try:
                response = self._llm.call_fast(
                    prompt=prompt,
                    system_prompt=SENTIMENT_SYSTEM_PROMPT,
                    temperature=0.3,
                )
                raw_content = response.get("content", "")
                provider = response.get("provider", "unknown")
                model = response.get("model", "unknown")

                parsed = _parse_json_list(raw_content)

                insert_agent_log(
                    event_type="sentiment_analysis",
                    content={
                        "batch_size": len(batch),
                        "provider": provider,
                        "model": model,
                        "results_count": len(parsed),
                        "results": parsed,
                    },
                )

                results.extend(parsed)

            except Exception as exc:
                logger.error(
                    "Sentiment analysis failed for batch starting at index %d: %s",
                    batch_start,
                    exc,
                )
                insert_agent_log(
                    event_type="sentiment_analysis_error",
                    content={"error": str(exc), "batch_size": len(batch)},
                )

        return results

    def make_decisions(
        self,
        portfolio_state: dict,
        sentiment_signals: list[dict],
        current_prices: dict[str, float],
        fundamentals: dict[str, dict],
        macro_snapshot: dict | None,
    ) -> list[TradeAction]:
        """Build context summaries, call LLM with decision prompt, parse JSON
        response into TradeAction objects."""
        portfolio_summary = _format_portfolio_summary(portfolio_state)
        sentiment_text = _format_sentiment_signals(sentiment_signals)
        price_text = _format_price_data(current_prices)
        fundamentals_text = _format_fundamentals(fundamentals)
        macro_text = _format_macro_summary(macro_snapshot)

        prompt = DECISION_USER_PROMPT.format(
            portfolio_summary=portfolio_summary,
            sentiment_signals=sentiment_text,
            price_data=price_text,
            fundamentals_summary=fundamentals_text,
            macro_summary=macro_text,
        )

        try:
            response = self._llm.call(
                prompt=prompt,
                system_prompt=DECISION_SYSTEM_PROMPT,
                temperature=0.3,
            )
            raw_content = response.get("content", "")
            provider = response.get("provider", "unknown")
            model = response.get("model", "unknown")

            parsed = _parse_json_list(raw_content)

            trades: list[TradeAction] = []
            for item in parsed:
                try:
                    trades.append(
                        TradeAction(
                            ticker=item["ticker"],
                            action=item["action"],
                            quantity=int(item.get("quantity", 0)),
                            reasoning=item.get("reasoning", ""),
                            confidence=float(item.get("confidence", 0.0)),
                            urgency=item.get("urgency", "low"),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed trade decision item %s: %s", item, exc
                    )

            insert_agent_log(
                event_type="trading_decisions",
                content={
                    "provider": provider,
                    "model": model,
                    "decisions_count": len(trades),
                    "decisions": [
                        {
                            "ticker": t.ticker,
                            "action": t.action,
                            "quantity": t.quantity,
                            "confidence": t.confidence,
                            "urgency": t.urgency,
                            "reasoning": t.reasoning,
                        }
                        for t in trades
                    ],
                },
            )

            return trades

        except Exception as exc:
            logger.error("make_decisions failed: %s", exc)
            insert_agent_log(
                event_type="trading_decisions_error",
                content={"error": str(exc)},
            )
            return []
