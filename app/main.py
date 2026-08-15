import logging
import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from arq.connections import create_pool, RedisSettings

from app.core.config import settings
from app.db.session import get_db, engine
from app.api.v1.router import api_v1_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} ({settings.APP_ENV})...")

    # Shared OpenAI client (one HTTP connection pool for the whole process)
    app.state.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Shared Redis client
    app.state.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Shared ARQ pool for enqueueing background jobs
    app.state.arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info(f"Shutting down {settings.APP_NAME}...")
    await app.state.openai_client.close()
    await app.state.redis.aclose()
    await app.state.arq_pool.aclose()
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-tenant AI Medical Assistant API based on TZ v1.0",
    version="1.0.0",
    lifespan=lifespan
)

# ── Middleware (must be added BEFORE routers) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=False,   # False is correct when origins are not "*"
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(api_v1_router)

# ── Static Admin Panel ────────────────────────────────────────────────────────
admin_dir = os.path.join(os.path.dirname(__file__), "static", "admin")
if os.path.exists(admin_dir):
    app.mount("/admin", StaticFiles(directory=admin_dir, html=True), name="admin")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Monitoring"])
async def health_check(db: AsyncSession = Depends(get_db)):
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "database": "unknown",
        "redis": "unknown",
        "latency_ms": 0
    }

    # Check Database
    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            health_status["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"

    # Check Redis (use shared pool from app state)
    try:
        r: redis.Redis = app.state.redis
        ping_ok = await r.ping()
        if ping_ok:
            health_status["redis"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["redis"] = f"error: {str(e)}"

    health_status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
    return health_status
