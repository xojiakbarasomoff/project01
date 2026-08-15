from fastapi import APIRouter
from app.api.v1.telegram import router as telegram_router
from app.api.v1.admin.appointments import router as appointments_router
from app.api.v1.admin.conversations import router as conversations_router
from app.api.v1.admin.knowledge_base import router as kb_router
from app.api.v1.admin.settings import router as settings_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(telegram_router)
api_v1_router.include_router(appointments_router)
api_v1_router.include_router(conversations_router)
api_v1_router.include_router(kb_router)
api_v1_router.include_router(settings_router)
