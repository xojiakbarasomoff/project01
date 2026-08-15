import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Doctor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/doctors", tags=["Admin — Doctors"])


class DoctorCreate(BaseModel):
    name: str
    specialty: str = "Stomatolog"
    phone: Optional[str] = None
    working_hours: str = "09:00 - 18:00"
    tenant_id: int = 1


class DoctorUpdate(BaseModel):
    name: Optional[str] = None
    specialty: Optional[str] = None
    phone: Optional[str] = None
    working_hours: Optional[str] = None
    is_active: Optional[bool] = None


class DoctorResponse(BaseModel):
    id: int
    tenant_id: int
    name: str
    specialty: str
    phone: Optional[str] = None
    working_hours: str
    is_active: bool

    class Config:
        from_attributes = True


@router.get("", response_model=List[DoctorResponse])
async def list_doctors(
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves all active doctors for clinic."""
    stmt = select(Doctor).where(Doctor.tenant_id == tenant_id, Doctor.is_active.is_(True)).order_by(Doctor.id.asc())
    res = await db.execute(stmt)
    return res.scalars().all()


@router.post("", response_model=DoctorResponse, status_code=status.HTTP_201_CREATED)
async def create_doctor(
    payload: DoctorCreate,
    db: AsyncSession = Depends(get_db)
):
    """Creates a new doctor record."""
    doctor = Doctor(
        tenant_id=payload.tenant_id,
        name=payload.name,
        specialty=payload.specialty,
        phone=payload.phone,
        working_hours=payload.working_hours,
        is_active=True
    )
    db.add(doctor)
    await db.commit()
    await db.refresh(doctor)
    return doctor


@router.patch("/{doctor_id}", response_model=DoctorResponse)
async def update_doctor(
    doctor_id: int,
    payload: DoctorUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Updates doctor information."""
    stmt = select(Doctor).where(Doctor.id == doctor_id)
    res = await db.execute(stmt)
    doctor = res.scalar_one_or_none()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    for field, val in payload.model_dump(exclude_unset=True).items():
        setattr(doctor, field, val)

    await db.commit()
    await db.refresh(doctor)
    return doctor


@router.delete("/{doctor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_doctor(
    doctor_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Deactivates a doctor record."""
    stmt = select(Doctor).where(Doctor.id == doctor_id)
    res = await db.execute(stmt)
    doctor = res.scalar_one_or_none()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    doctor.is_active = False
    await db.commit()
    return None
