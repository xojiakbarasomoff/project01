from fastapi import APIRouter
from app.api.v1.telegram import router as telegram_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(telegram_router)
