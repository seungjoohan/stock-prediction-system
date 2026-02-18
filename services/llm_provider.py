import logging
import time
from datetime import datetime, date

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_FAST_MODEL,
    GOOGLE_GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Track API usage per provider."""

    def __init__(self, max_rpm: int, max_daily: int):
        self.max_rpm = max_rpm
        self.max_daily = max_daily
        self._minute_calls = 0
        self._daily_calls = 0
        self._current_minute = datetime.now().minute
        self._current_day = date.today()

    def reset_if_needed(self):
        """Reset counters at minute/day boundaries."""
        now = datetime.now()
        today = date.today()

        if now.minute != self._current_minute:
            self._minute_calls = 0
            self._current_minute = now.minute

        if today != self._current_day:
            self._daily_calls = 0
            self._current_day = today

    def can_call(self) -> bool:
        """Check if we're within rate limits."""
        self.reset_if_needed()
        return (
            self._minute_calls < self.max_rpm
            and self._daily_calls < self.max_daily
        )

    def record_call(self):
        """Record a call was made."""
        self.reset_if_needed()
        self._minute_calls += 1
        self._daily_calls += 1


class LLMProvider:
    def __init__(self):
        """Initialize Groq and Gemini clients with rate limiters."""
        self.groq_client = None
        self.gemini_client = None
        # Two separate rate limiters: large model (100k TPD) and fast model (500k TPD)
        self.groq_limiter = RateLimiter(max_rpm=30, max_daily=14400)
        self.groq_fast_limiter = RateLimiter(max_rpm=30, max_daily=72000)
        self.gemini_limiter = RateLimiter(max_rpm=15, max_daily=1000000)

        if GROQ_API_KEY:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialized.")
            except Exception as exc:
                logger.warning("Failed to initialize Groq client: %s", exc)

        if GOOGLE_GEMINI_API_KEY:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)
                logger.info("Gemini client initialized.")
            except Exception as exc:
                logger.warning("Failed to initialize Gemini client: %s", exc)

    def call(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """Call with the large model (decisions). Falls back to Gemini."""
        providers = [
            ("groq", self.groq_client, self.groq_limiter, GROQ_MODEL,
             lambda p, s, t: self._call_groq(p, s, t, GROQ_MODEL)),
            ("gemini", self.gemini_client, self.gemini_limiter, GEMINI_MODEL, self._call_gemini),
        ]
        return self._dispatch(prompt, system_prompt, temperature, providers)

    def call_fast(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """Call with the fast/small model (sentiment). Falls back to large, then Gemini."""
        providers = [
            ("groq-fast", self.groq_client, self.groq_fast_limiter, GROQ_FAST_MODEL,
             lambda p, s, t: self._call_groq(p, s, t, GROQ_FAST_MODEL)),
            ("groq", self.groq_client, self.groq_limiter, GROQ_MODEL,
             lambda p, s, t: self._call_groq(p, s, t, GROQ_MODEL)),
            ("gemini", self.gemini_client, self.gemini_limiter, GEMINI_MODEL, self._call_gemini),
        ]
        return self._dispatch(prompt, system_prompt, temperature, providers)

    def _dispatch(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        providers: list,
    ) -> dict:
        """Shared dispatch logic with per-provider retry and fallback."""

        last_error = None

        for provider_name, client, limiter, model, call_fn in providers:
            if client is None:
                logger.debug("Provider %s not initialized, skipping.", provider_name)
                continue

            if not limiter.can_call():
                logger.warning(
                    "Provider %s rate limit reached, skipping.", provider_name
                )
                continue

            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    limiter.record_call()
                    content = call_fn(prompt, system_prompt, temperature)
                    return {
                        "content": content,
                        "provider": provider_name,
                        "model": model,
                    }
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Provider %s attempt %d/%d failed: %s",
                        provider_name,
                        attempt,
                        LLM_MAX_RETRIES,
                        exc,
                    )
                    if attempt < LLM_MAX_RETRIES:
                        backoff = 2 ** (attempt - 1)
                        logger.debug("Backing off %ds before retry.", backoff)
                        time.sleep(backoff)

            logger.warning(
                "Provider %s exhausted after %d retries, falling through.",
                provider_name,
                LLM_MAX_RETRIES,
            )

        raise RuntimeError(
            f"All LLM providers exhausted. Last error: {last_error}"
        )

    def _call_groq(
        self, prompt: str, system_prompt: str, temperature: float, model: str = GROQ_MODEL
    ) -> str:
        """Call Groq API with the specified model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=LLM_TIMEOUT,
        )
        return response.choices[0].message.content

    def _call_gemini(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call Google Gemini API using the new google-genai SDK."""
        from google.genai import types

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        return response.text

    def get_usage_stats(self) -> dict:
        """Return current usage stats for all providers."""
        self.groq_limiter.reset_if_needed()
        self.groq_fast_limiter.reset_if_needed()
        self.gemini_limiter.reset_if_needed()

        return {
            "groq": {
                "model": GROQ_MODEL,
                "minute_calls": self.groq_limiter._minute_calls,
                "daily_calls": self.groq_limiter._daily_calls,
                "max_rpm": self.groq_limiter.max_rpm,
                "max_daily": self.groq_limiter.max_daily,
                "available": self.groq_client is not None,
            },
            "groq-fast": {
                "model": GROQ_FAST_MODEL,
                "minute_calls": self.groq_fast_limiter._minute_calls,
                "daily_calls": self.groq_fast_limiter._daily_calls,
                "max_rpm": self.groq_fast_limiter.max_rpm,
                "max_daily": self.groq_fast_limiter.max_daily,
                "available": self.groq_client is not None,
            },
            "gemini": {
                "model": GEMINI_MODEL,
                "minute_calls": self.gemini_limiter._minute_calls,
                "daily_calls": self.gemini_limiter._daily_calls,
                "max_rpm": self.gemini_limiter.max_rpm,
                "max_daily": self.gemini_limiter.max_daily,
                "available": self.gemini_client is not None,
            },
        }
