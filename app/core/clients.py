"""
app/core/clients.py
-------------------
Thin accessors for process-wide singletons (OpenAI, Redis, ARQ pool).

These objects are created once in the FastAPI lifespan (main.py) and stored on
``app.state``.  Workers that run outside the FastAPI process (ARQ tasks) call
``get_openai_client()`` / ``get_redis_client()`` which lazily create their own
pooled instances so they still work without the web server.
"""
from __future__ import annotations

import redis.asyncio as aioredis
from arq.connections import ArqRedis, RedisSettings, create_pool
from openai import AsyncOpenAI

from app.core.config import settings

# Module-level fallback singletons for the worker process.
# In the web process these are *never* used — the lifespan-managed instances
# on app.state are injected directly.
_openai_client: AsyncOpenAI | None = None
_redis_client: aioredis.Redis | None = None
_arq_pool: ArqRedis | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return the process-wide AsyncOpenAI client, creating it if necessary."""
    global _openai_client
    if _openai_client is None:
        kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        elif (
            settings.OPENAI_API_KEY.startswith("AQ.")
            or settings.OPENAI_API_KEY.startswith("AIza")
            or "gemini" in settings.OPENAI_MODEL.lower()
        ):
            kwargs["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        _openai_client = AsyncOpenAI(**kwargs)
    return _openai_client


async def get_redis_client() -> aioredis.Redis:
    """Return the process-wide async Redis client, creating it if necessary."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


async def get_arq_pool() -> ArqRedis:
    """Return the process-wide ARQ pool, creating it if necessary."""
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    return _arq_pool
