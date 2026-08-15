import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/settings", tags=["Admin — Tenant Settings"])


class TenantSettingsUpdate(BaseModel):
    debounce_seconds: Optional[int] = Field(None, ge=1, le=300, description="Debounce wait time in seconds (1-300)")


@router.get("")
async def get_tenant_settings(
    tenant_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Retrieves current tenant settings."""
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    res = await db.execute(stmt)
    tenant = res.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    settings_dict = tenant.settings if isinstance(tenant.settings, dict) else {}
    return {
        "tenant_id": tenant.id,
        "name": tenant.name,
        "debounce_seconds": settings_dict.get("debounce_seconds", 30),
        "settings": settings_dict
    }


@router.post("")
async def update_tenant_settings(
    payload: TenantSettingsUpdate,
    tenant_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Updates tenant settings (such as debounce wait time)."""
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    res = await db.execute(stmt)
    tenant = res.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    current_settings = dict(tenant.settings) if isinstance(tenant.settings, dict) else {}

    if payload.debounce_seconds is not None:
        current_settings["debounce_seconds"] = payload.debounce_seconds

    tenant.settings = current_settings
    await db.commit()
    await db.refresh(tenant)

    logger.info(f"⚙️ Tenant {tenant_id} settings updated: {current_settings}")

    return {
        "status": "success",
        "debounce_seconds": current_settings.get("debounce_seconds", 30),
        "settings": current_settings
    }
