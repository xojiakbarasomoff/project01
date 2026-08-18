from fastapi import APIRouter, Depends
from app.api.v1.telegram import router as telegram_router
from app.api.v1.webapp import router as webapp_router
from app.api.v1.admin.auth import router as auth_router, get_current_admin
from app.api.v1.admin.appointments import router as appointments_router
from app.api.v1.admin.conversations import router as conversations_router
from app.api.v1.admin.knowledge_base import router as kb_router
from app.api.v1.admin.settings import router as settings_router
from app.api.v1.admin.analytics import router as analytics_router
from app.api.v1.admin.doctors import router as doctors_router
from app.api.v1.admin.export import router as export_router
from app.api.v1.admin.leads import router as leads_router
from app.api.v1.admin.strict_rules import router as strict_rules_router

api_v1_router = APIRouter(prefix="/api/v1")

# Public routers
api_v1_router.include_router(telegram_router)
api_v1_router.include_router(webapp_router)

# Unprotected Auth router
api_v1_router.include_router(auth_router, prefix="/admin")

# Protected Admin endpoints (requires Authorization header)
admin_protected_router = APIRouter(dependencies=[Depends(get_current_admin)])
admin_protected_router.include_router(appointments_router)
admin_protected_router.include_router(conversations_router)
admin_protected_router.include_router(kb_router)
admin_protected_router.include_router(settings_router)
admin_protected_router.include_router(analytics_router)
admin_protected_router.include_router(doctors_router)
admin_protected_router.include_router(export_router)
admin_protected_router.include_router(leads_router)
admin_protected_router.include_router(strict_rules_router)

api_v1_router.include_router(admin_protected_router)

