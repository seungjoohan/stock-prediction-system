# Stock Prediction System

Started as a personal ML learning project (v1): a Streamlit app that pulled stock data via yfinance, trained ARIMA and Prophet models on TSLA/AAPL/GOOG, and served predictions. The focus was on ML system architecture and model production over prediction accuracy. v2 rebuilt it into a full agentic trading system on top of the same repo.

---

## Architecture (v2)

```
                          [ Market Hours Loop ]
                                   |
              every 5 min          |          every 15 min
         .-------------------.     |     .--------------------.
         | News ingestion    |     |     | Decision cycle     |
         | - Finnhub API     |     |     | 1. Gather context  |
         | - RSS feeds       |     |     |    - Live prices   |
         | LLM sentiment     |     |     |    - Portfolio     |
         |   (Groq/Gemini)   |     |     |    - FRED macro    |
         | RAG indexing      |     |     |    - Fundamentals  |
         |   - SQLite insert |     |     |    - Sentiment     |
         |   - ChromaDB      |     |     |    - RAG history   |
         '-------------------'     |     | 2. Build prompt    |
                                   |     | 3. LLM decision    |
                                   |     | 4. Risk validation |
                                   |     | 5. Execute trades  |
                                   |     |    via Alpaca API  |
                                   |     '--------------------'
                                   |
                          [ SQLite + ChromaDB ]
                          (persistent across restarts)
```

### Directory layout

```
agent/            Core agent loop, decision engine, risk manager, prompts
config/           Settings: API keys, tracked symbols, constants
dashboard/        Streamlit monitoring dashboard
db/               SQLite (trading.db, WAL mode) + ChromaDB vector store
services/         Market data, news, fundamentals, macro, LLM, RAG, trade executor
tests/            88 unit tests
predict_stocks.py v1 Streamlit app (still works)
```

### Key design points

- **Two-layer RAG**: SQL for structured retrieval (7-day window) + ChromaDB for semantic similarity search — memory persists across restarts. Falls back to SQL-only if ChromaDB/embeddings are unavailable.
- **LLM**: Groq (llama/mixtral) primary, Google Gemini fallback. Rate limiter counts only successful calls.
- **Risk guards**: confidence >= 0.65, max 10 buys/day, daily loss circuit breaker (sells always allowed for loss-cutting), VIX < 35, position size <= 5% of portfolio.
- **Paper vs. live**: controlled by `ALPACA_PAPER_TRADING` env var (`false` = live).
- **20 tracked US equity symbols.**

### APIs

| Service       | Purpose                              |
| ------------- | ------------------------------------ |
| Alpaca        | Trade execution, portfolio sync      |
| Finnhub       | News, fundamentals, WebSocket prices |
| FRED          | Macro data snapshots                 |
| Groq          | Primary LLM                          |
| Google Gemini | LLM fallback                         |

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root, fill out your api keys and configurations:

```
cp .env.example .env
```

---

## Running

```bash
# v2 agentic trading system
python -m agent.core

# v1 Streamlit prediction app
streamlit run predict_stocks.py

# Monitoring dashboard
streamlit run dashboard/app.py
```

---

## Tests

```bash
python -m pytest tests/ -q
# 88 passed
```

---

## v3 — Planned: Hybrid Quantitative + RAG

The current v2 system relies entirely on LLM judgment for trading decisions, grounded by sentiment signals and historical news context via RAG. v3 aims to close the loop by incorporating quantitative model outputs as an additional signal layer.

**Core idea**: run quantitative methods (statistical models, ML-based forecasts, technical indicators, etc.) on price and market data to generate structured signals, then feed those into the existing RAG + LLM decision pipeline alongside news sentiment. The LLM would reason over both the quantitative signal ("model expects upward move with 78% confidence") and the qualitative context ("recent bearish sentiment, similar news historically preceded a reversal") before deciding.

**Rough direction:**
- Quantitative models produce per-ticker signals (direction, magnitude, confidence)
- Signals stored persistently and retrieved as a third RAG layer at decision time
- LLM reconciles quantitative predictions with sentiment — agreeing signals reinforce conviction, conflicting signals trigger conservative sizing or a hold
- Track signal accuracy over time to weight models dynamically
