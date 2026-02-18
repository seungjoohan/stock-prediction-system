import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fredapi import Fred

from config.settings import FRED_API_KEY
from db.database import insert_macro_snapshot, get_latest_macro_snapshot

logger = logging.getLogger(__name__)


@dataclass
class MacroSnapshot:
    timestamp: str
    fed_funds_rate: float | None
    treasury_10y: float | None
    treasury_2y: float | None
    yield_curve_spread: float | None
    cpi_yoy: float | None
    unemployment_rate: float | None
    gdp_growth_qoq: float | None
    vix: float | None
    consumer_sentiment: float | None
    initial_jobless_claims: float | None
    derived_signals: dict = field(default_factory=dict)


class MacroDataService:
    def __init__(self) -> None:
        self.fred = Fred(api_key=FRED_API_KEY)

    def _fetch_last_value(self, series_id: str) -> float | None:
        try:
            series = self.fred.get_series(series_id)
            non_nan = series.dropna()
            if non_nan.empty:
                logger.warning("Series %s returned no non-NaN values", series_id)
                return None
            return float(non_nan.iloc[-1])
        except Exception as exc:
            logger.warning("Failed to fetch FRED series %s: %s", series_id, exc)
            return None

    def _fetch_cpi_yoy(self) -> float | None:
        try:
            series = self.fred.get_series("CPIAUCSL")
            non_nan = series.dropna()
            if len(non_nan) < 13:
                logger.warning("CPIAUCSL series has fewer than 13 observations; cannot compute YoY")
                return None
            latest = float(non_nan.iloc[-1])
            year_ago = float(non_nan.iloc[-13])
            return ((latest - year_ago) / year_ago) * 100.0
        except Exception as exc:
            logger.warning("Failed to compute CPI YoY: %s", exc)
            return None

    def refresh_macro_data(self) -> MacroSnapshot:
        fed_funds_rate = self._fetch_last_value("FEDFUNDS")
        treasury_10y = self._fetch_last_value("DGS10")
        treasury_2y = self._fetch_last_value("DGS2")
        cpi_yoy = self._fetch_cpi_yoy()
        unemployment_rate = self._fetch_last_value("UNRATE")
        gdp_growth_qoq = self._fetch_last_value("GDP")
        vix = self._fetch_last_value("VIXCLS")
        consumer_sentiment = self._fetch_last_value("UMCSENT")
        initial_jobless_claims = self._fetch_last_value("ICSA")

        if treasury_10y is not None and treasury_2y is not None:
            yield_curve_spread = treasury_10y - treasury_2y
        else:
            yield_curve_spread = None

        timestamp = datetime.now(timezone.utc).isoformat()

        snapshot = MacroSnapshot(
            timestamp=timestamp,
            fed_funds_rate=fed_funds_rate,
            treasury_10y=treasury_10y,
            treasury_2y=treasury_2y,
            yield_curve_spread=yield_curve_spread,
            cpi_yoy=cpi_yoy,
            unemployment_rate=unemployment_rate,
            gdp_growth_qoq=gdp_growth_qoq,
            vix=vix,
            consumer_sentiment=consumer_sentiment,
            initial_jobless_claims=initial_jobless_claims,
        )
        snapshot.derived_signals = self.get_derived_signals(snapshot)

        insert_macro_snapshot({
            "timestamp": snapshot.timestamp,
            "fed_funds_rate": snapshot.fed_funds_rate,
            "treasury_10y": snapshot.treasury_10y,
            "treasury_2y": snapshot.treasury_2y,
            "yield_curve_spread": snapshot.yield_curve_spread,
            "cpi_yoy": snapshot.cpi_yoy,
            "unemployment_rate": snapshot.unemployment_rate,
            "gdp_growth_qoq": snapshot.gdp_growth_qoq,
            "vix": snapshot.vix,
            "consumer_sentiment": snapshot.consumer_sentiment,
            "initial_jobless_claims": snapshot.initial_jobless_claims,
        })

        logger.info("Macro snapshot refreshed at %s", timestamp)
        return snapshot

    def get_macro_snapshot(self) -> MacroSnapshot | None:
        row = get_latest_macro_snapshot()
        if row is None:
            return None

        snapshot = MacroSnapshot(
            timestamp=row.get("timestamp", ""),
            fed_funds_rate=row.get("fed_funds_rate"),
            treasury_10y=row.get("treasury_10y"),
            treasury_2y=row.get("treasury_2y"),
            yield_curve_spread=row.get("yield_curve_spread"),
            cpi_yoy=row.get("cpi_yoy"),
            unemployment_rate=row.get("unemployment_rate"),
            gdp_growth_qoq=row.get("gdp_growth_qoq"),
            vix=row.get("vix"),
            consumer_sentiment=row.get("consumer_sentiment"),
            initial_jobless_claims=row.get("initial_jobless_claims"),
        )
        snapshot.derived_signals = self.get_derived_signals(snapshot)
        return snapshot

    def get_derived_signals(self, snapshot: MacroSnapshot) -> dict:
        return {
            "yield_curve_inverted": (
                snapshot.yield_curve_spread < 0
                if snapshot.yield_curve_spread is not None
                else False
            ),
            "vix_elevated": (
                snapshot.vix > 25
                if snapshot.vix is not None
                else False
            ),
            "vix_extreme": (
                snapshot.vix > 35
                if snapshot.vix is not None
                else False
            ),
            "inflation_hot": (
                snapshot.cpi_yoy > 3.5
                if snapshot.cpi_yoy is not None
                else False
            ),
        }
