import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import finnhub

from config.settings import FINNHUB_API_KEY, TRACKED_SYMBOLS
from db.database import get_company_fundamentals, upsert_company_fundamentals

logger = logging.getLogger(__name__)


@dataclass
class CompanyFundamentals:
    ticker: str
    updated_at: datetime

    # Valuation
    pe_ttm: float | None
    forward_pe: float | None
    ps_ratio: float | None
    pb_ratio: float | None
    ev_ebitda: float | None

    # Profitability
    gross_margin: float | None
    operating_margin: float | None
    roe: float | None
    roa: float | None

    # Growth
    revenue_growth_yoy: float | None
    eps_growth_yoy: float | None

    # Earnings
    last_eps_actual: float | None
    last_eps_estimate: float | None
    earnings_surprise_pct: float | None
    next_earnings_date: date | None

    # Analyst consensus
    analyst_buy: int
    analyst_hold: int
    analyst_sell: int
    avg_price_target: float | None

    # Dividend
    dividend_yield: float | None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


class FundamentalsService:
    def __init__(self):
        self._client = finnhub.Client(api_key=FINNHUB_API_KEY)

    def refresh_fundamentals(self, ticker: str) -> CompanyFundamentals:
        metrics_data = self._client.company_basic_financials(ticker, "all")
        metric = metrics_data.get("metric", {}) if metrics_data else {}

        earnings_data = self._client.company_earnings(ticker, limit=1)
        last_eps_actual: float | None = None
        last_eps_estimate: float | None = None
        earnings_surprise_pct: float | None = None
        if earnings_data:
            latest = earnings_data[0]
            last_eps_actual = _safe_float(latest.get("actual"))
            last_eps_estimate = _safe_float(latest.get("estimate"))
            surprise = _safe_float(latest.get("surprisePercent"))
            if surprise is None and last_eps_estimate and last_eps_actual is not None:
                if last_eps_estimate != 0:
                    surprise = (last_eps_actual - last_eps_estimate) / abs(last_eps_estimate) * 100
            earnings_surprise_pct = surprise

        rec_data = self._client.recommendation_trends(ticker)
        analyst_buy = 0
        analyst_hold = 0
        analyst_sell = 0
        if rec_data:
            latest_rec = rec_data[0]
            analyst_buy = int(latest_rec.get("buy", 0) or 0)
            analyst_hold = int(latest_rec.get("hold", 0) or 0)
            analyst_sell = int(latest_rec.get("sell", 0) or 0)

        avg_price_target: float | None = None
        try:
            target_data = self._client.price_target(ticker)
            if target_data:
                avg_price_target = _safe_float(target_data.get("targetMean"))
        except Exception as exc:
            logger.debug("Could not fetch price target for %s (may require premium): %s", ticker, exc)

        next_earnings_date: date | None = None
        try:
            calendar_data = self._client.earnings_calendar(
                _from="", to="", symbol=ticker, international=False
            )
            earnings_list = (calendar_data or {}).get("earningsCalendar", [])
            today = date.today()
            future_dates = [
                _parse_date(e.get("date"))
                for e in earnings_list
                if _parse_date(e.get("date")) and _parse_date(e.get("date")) >= today
            ]
            if future_dates:
                next_earnings_date = min(future_dates)
        except Exception as exc:
            logger.warning("Could not fetch earnings calendar for %s: %s", ticker, exc)

        fundamentals = CompanyFundamentals(
            ticker=ticker,
            updated_at=datetime.now(timezone.utc),
            pe_ttm=_safe_float(metric.get("peTTM")),
            forward_pe=_safe_float(metric.get("peForward")),
            ps_ratio=_safe_float(metric.get("psTTM")),
            pb_ratio=_safe_float(metric.get("pbAnnual")),
            ev_ebitda=_safe_float(metric.get("currentEv/freeCashFlowAnnual")),
            gross_margin=_safe_float(metric.get("grossMarginTTM")),
            operating_margin=_safe_float(metric.get("operatingMarginTTM")),
            roe=_safe_float(metric.get("roeTTM")),
            roa=_safe_float(metric.get("roaTTM")),
            revenue_growth_yoy=_safe_float(metric.get("revenueGrowthTTMYoy")),
            eps_growth_yoy=_safe_float(metric.get("epsGrowthTTMYoy")),
            last_eps_actual=last_eps_actual,
            last_eps_estimate=last_eps_estimate,
            earnings_surprise_pct=earnings_surprise_pct,
            next_earnings_date=next_earnings_date,
            analyst_buy=analyst_buy,
            analyst_hold=analyst_hold,
            analyst_sell=analyst_sell,
            avg_price_target=avg_price_target,
            dividend_yield=_safe_float(metric.get("dividendYieldIndicatedAnnual")),
        )

        upsert_company_fundamentals({
            "ticker": fundamentals.ticker,
            "pe_ttm": fundamentals.pe_ttm,
            "forward_pe": fundamentals.forward_pe,
            "ps_ratio": fundamentals.ps_ratio,
            "pb_ratio": fundamentals.pb_ratio,
            "ev_ebitda": fundamentals.ev_ebitda,
            "gross_margin": fundamentals.gross_margin,
            "operating_margin": fundamentals.operating_margin,
            "roe": fundamentals.roe,
            "roa": fundamentals.roa,
            "revenue_growth_yoy": fundamentals.revenue_growth_yoy,
            "eps_growth_yoy": fundamentals.eps_growth_yoy,
            "last_eps_actual": fundamentals.last_eps_actual,
            "last_eps_estimate": fundamentals.last_eps_estimate,
            "earnings_surprise_pct": fundamentals.earnings_surprise_pct,
            "next_earnings_date": (
                fundamentals.next_earnings_date.isoformat()
                if fundamentals.next_earnings_date
                else None
            ),
            "analyst_buy": fundamentals.analyst_buy,
            "analyst_hold": fundamentals.analyst_hold,
            "analyst_sell": fundamentals.analyst_sell,
            "avg_price_target": fundamentals.avg_price_target,
            "dividend_yield": fundamentals.dividend_yield,
        })

        logger.info("Refreshed fundamentals for %s", ticker)
        return fundamentals

    def refresh_all(self) -> list[CompanyFundamentals]:
        results: list[CompanyFundamentals] = []
        for ticker in TRACKED_SYMBOLS:
            try:
                fundamentals = self.refresh_fundamentals(ticker)
                results.append(fundamentals)
            except Exception as exc:
                logger.warning("Failed to refresh fundamentals for %s: %s", ticker, exc)
        return results

    def get_fundamentals(self, ticker: str) -> CompanyFundamentals | None:
        row = get_company_fundamentals(ticker)
        if row is None:
            return None
        return CompanyFundamentals(
            ticker=row["ticker"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            pe_ttm=row.get("pe_ttm"),
            forward_pe=row.get("forward_pe"),
            ps_ratio=row.get("ps_ratio"),
            pb_ratio=row.get("pb_ratio"),
            ev_ebitda=row.get("ev_ebitda"),
            gross_margin=row.get("gross_margin"),
            operating_margin=row.get("operating_margin"),
            roe=row.get("roe"),
            roa=row.get("roa"),
            revenue_growth_yoy=row.get("revenue_growth_yoy"),
            eps_growth_yoy=row.get("eps_growth_yoy"),
            last_eps_actual=row.get("last_eps_actual"),
            last_eps_estimate=row.get("last_eps_estimate"),
            earnings_surprise_pct=row.get("earnings_surprise_pct"),
            next_earnings_date=_parse_date(row.get("next_earnings_date")),
            analyst_buy=int(row.get("analyst_buy") or 0),
            analyst_hold=int(row.get("analyst_hold") or 0),
            analyst_sell=int(row.get("analyst_sell") or 0),
            avg_price_target=row.get("avg_price_target"),
            dividend_yield=row.get("dividend_yield"),
        )
