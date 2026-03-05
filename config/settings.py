import os
from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- Agent Settings ---
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
MIN_TRADE_CONFIDENCE = float(os.getenv("MIN_TRADE_CONFIDENCE", "0.6"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
CASH_RESERVE_PCT = float(os.getenv("CASH_RESERVE_PCT", "0.05"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
DECISION_INTERVAL_MIN = int(os.getenv("DECISION_INTERVAL_MIN", "15"))
NEWS_POLL_INTERVAL_MIN = int(os.getenv("NEWS_POLL_INTERVAL_MIN", "5"))

# --- Tracked Symbols ---
_default_symbols = (
    "AAPL,MSFT,GOOG,AMZN,NVDA,META,TSLA,JPM,UNH,JNJ,"
    "XOM,PG,HD,BA,DIS,AMD,COST,CRM,SPY,QQQ"
)
TRACKED_SYMBOLS = os.getenv("TRACKED_SYMBOLS", _default_symbols).split(",")

# --- Database ---
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "db", "trading.db"))

# --- Risk Thresholds ---
VIX_CIRCUIT_BREAKER = float(os.getenv("VIX_CIRCUIT_BREAKER", "35"))
EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "2"))
SIGNIFICANT_MOVE_PCT = float(os.getenv("SIGNIFICANT_MOVE_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.08"))       # force-sell if position down ≥8%

# --- Short Selling ---
ALLOW_SHORT_SELLING    = os.getenv("ALLOW_SHORT_SELLING", "false").lower() == "true"
MAX_SHORT_POSITIONS    = int(os.getenv("MAX_SHORT_POSITIONS", "5"))
MAX_SHORT_EXPOSURE_PCT = float(os.getenv("MAX_SHORT_EXPOSURE_PCT", "0.10"))
MIN_SHORT_CONFIDENCE   = float(os.getenv("MIN_SHORT_CONFIDENCE", "0.80"))
MAX_SHORT_POSITION_PCT = float(os.getenv("MAX_SHORT_POSITION_PCT", "0.05"))

# --- Drawdown Circuit Breaker ---
MAX_DRAWDOWN_PCT    = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))
DRAWDOWN_HALT_HOURS = int(os.getenv("DRAWDOWN_HALT_HOURS", "48"))

# --- Sector Exposure ---
MAX_SECTOR_EXPOSURE_PCT = float(os.getenv("MAX_SECTOR_EXPOSURE_PCT", "0.40"))
SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Technology",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "XOM": "Energy",
    "PG": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "BA": "Industrials",
    "DIS": "Communication Services",
    "AMD": "Technology",
    "COST": "Consumer Staples",
    "CRM": "Technology",
    "SPY": "ETF",
    "QQQ": "ETF",
}

# --- Finnhub WebSocket ---
FINNHUB_WS_URL = "wss://ws.finnhub.io"

# --- LLM Settings ---
# Large model: trading decisions (complex reasoning, low volume, 100k TPD)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Fast model: sentiment analysis (structured JSON, high volume, 500k TPD)
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# --- Market Hours (ET) ---
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# --- RSS Feed URLs ---
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
]
