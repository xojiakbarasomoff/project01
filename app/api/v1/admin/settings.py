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
    clinic_address: Optional[str] = Field(None, description="Clinic street address")
    clinic_landmark: Optional[str] = Field(None, description="Clinic landmark / moljal")
    clinic_latitude: Optional[float] = Field(None, description="Clinic GPS latitude")
    clinic_longitude: Optional[float] = Field(None, description="Clinic GPS longitude")
    clinic_work_hours: Optional[str] = Field(None, description="Clinic working hours")


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
        "clinic_address": settings_dict.get("clinic_address", "Toshkent shahri, Amir Temur shoh ko'chasi, 45-uy"),
        "clinic_landmark": settings_dict.get("clinic_landmark", "Markaziy Universitet qarshisida"),
        "clinic_latitude": float(settings_dict.get("clinic_latitude", 41.311081)),
        "clinic_longitude": float(settings_dict.get("clinic_longitude", 69.240562)),
        "clinic_work_hours": settings_dict.get("clinic_work_hours", "Har kuni 09:00 - 18:00"),
        "settings": settings_dict
    }


@router.post("")
async def update_tenant_settings(
    payload: TenantSettingsUpdate,
    tenant_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Updates tenant settings (including location, coordinates, and debounce)."""
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    res = await db.execute(stmt)
    tenant = res.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    current_settings = dict(tenant.settings) if isinstance(tenant.settings, dict) else {}

    if payload.debounce_seconds is not None:
        current_settings["debounce_seconds"] = payload.debounce_seconds
    if payload.clinic_address is not None:
        current_settings["clinic_address"] = payload.clinic_address
    if payload.clinic_landmark is not None:
        current_settings["clinic_landmark"] = payload.clinic_landmark
    if payload.clinic_latitude is not None:
        current_settings["clinic_latitude"] = payload.clinic_latitude
    if payload.clinic_longitude is not None:
        current_settings["clinic_longitude"] = payload.clinic_longitude
    if payload.clinic_work_hours is not None:
        current_settings["clinic_work_hours"] = payload.clinic_work_hours

    tenant.settings = current_settings
    await db.commit()
    await db.refresh(tenant)

    logger.info(f"⚙️ Tenant {tenant_id} settings updated: {current_settings}")

    return {
        "status": "success",
        "debounce_seconds": current_settings.get("debounce_seconds", 30),
        "clinic_address": current_settings.get("clinic_address", "Toshkent shahri, Amir Temur shoh ko'chasi, 45-uy"),
        "clinic_landmark": current_settings.get("clinic_landmark", "Markaziy Universitet qarshisida"),
        "clinic_latitude": float(current_settings.get("clinic_latitude", 41.311081)),
        "clinic_longitude": float(current_settings.get("clinic_longitude", 69.240562)),
        "clinic_work_hours": current_settings.get("clinic_work_hours", "Har kuni 09:00 - 18:00"),
        "settings": current_settings
    }
