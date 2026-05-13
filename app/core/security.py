from fastapi import Request, HTTPException, status, WebSocket
from fastapi.security import APIKeyHeader
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
import threading
import time

from app.core.config import settings

# ── Header scheme ─────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── In-memory rate limiter (token bucket per API key) ─────────────────────────
class RateLimiter:
    """
    Simple sliding-window rate limiter.
    Thread-safe enough for single-process deployments.
    For multi-process: replace with Redis.
    """

    def __init__(self, max_calls: int, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._records: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            calls = self._records[key]
            # Remove expired timestamps
            self._records[key] = [t for t in calls if t > cutoff]
            if len(self._records[key]) >= self.max_calls:
                return False
            self._records[key].append(now)
            return True

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            calls = [t for t in self._records[key] if t > cutoff]
            return max(0, self.max_calls - len(calls))


rate_limiter = RateLimiter(
    max_calls=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60,
)


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_api_key(api_key: str | None) -> str:
    """Raise 401/429 or return the validated key."""
    if not api_key or api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {settings.RATE_LIMIT_PER_MINUTE} req/min.",
            headers={"Retry-After": "60"},
        )
    return api_key


async def validate_ws_api_key(websocket: WebSocket) -> str:
    """Extract and validate API key from WebSocket query param."""
    api_key = websocket.query_params.get("api_key")
    if not api_key or api_key not in settings.api_key_set:
        await websocket.close(code=4001, reason="Unauthorized: invalid API key")
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not rate_limiter.is_allowed(api_key):
        await websocket.close(code=4029, reason="Rate limit exceeded")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return api_key
