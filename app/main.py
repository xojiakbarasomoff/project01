import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from app.core.config import settings
from app.db.session import get_db, engine
from app.api.v1.router import api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print(f"🚀 Starting {settings.APP_NAME} ({settings.APP_ENV})...")
    yield
    # Shutdown logic
    print(f"🛑 Shutting down {settings.APP_NAME}...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-tenant AI Medical Assistant API based on TZ v1.0",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_v1_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Check Database connection
    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            health_status["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"

    # Check Redis connection
    try:
        r = redis.from_url(settings.REDIS_URL)
        ping_ok = await r.ping()
        await r.close()
        if ping_ok:
            health_status["redis"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["redis"] = f"error: {str(e)}"

    health_status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
    return health_status
