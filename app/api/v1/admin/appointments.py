import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Appointment
from app.schemas.appointment import AppointmentCreate, AppointmentUpdate, AppointmentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/appointments", tags=["Admin — Appointments"])


@router.get("", response_model=List[AppointmentResponse])
async def list_appointments(
    tenant_id: int = Query(1, description="Tenant ID"),
    status: Optional[str] = Query(None, description="Filter by status (pending, confirmed, cancelled)"),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves all appointments for a clinic tenant."""
    stmt = select(Appointment).where(Appointment.tenant_id == tenant_id)
    if status:
        stmt = stmt.where(Appointment.status == status)

    stmt = stmt.order_by(desc(Appointment.created_at))
    res = await db.execute(stmt)
    return res.scalars().all()


@router.post("", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED)
async def create_appointment(
    payload: AppointmentCreate,
    db: AsyncSession = Depends(get_db)
):
    """Creates a new patient appointment record."""
    appointment = Appointment(
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        patient_name=payload.patient_name,
        patient_phone=payload.patient_phone,
        doctor_name=payload.doctor_name,
        appointment_time=payload.appointment_time,
        notes=payload.notes,
        status="pending"
    )
    db.add(appointment)
    await db.commit()
    await db.refresh(appointment)
    return appointment


@router.patch("/{appointment_id}", response_model=AppointmentResponse)
async def update_appointment(
    appointment_id: int,
    payload: AppointmentUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Updates status, notes, or date for an appointment."""
    stmt = select(Appointment).where(Appointment.id == appointment_id)
    res = await db.execute(stmt)
    appointment = res.scalar_one_or_none()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(appointment, field, val)

    await db.commit()
    await db.refresh(appointment)
    return appointment


@router.delete("/{appointment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_appointment(
    appointment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Deletes an appointment."""
    stmt = select(Appointment).where(Appointment.id == appointment_id)
    res = await db.execute(stmt)
    appointment = res.scalar_one_or_none()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    await db.delete(appointment)
    await db.commit()
    return None


@router.post("/run-reminders")
async def run_reminders_now():
    """Manually trigger appointment reminder checks (24h and 2h reminders)."""
    from app.worker.tasks import check_appointment_reminders
    sent = await check_appointment_reminders()
    return {"status": "success", "reminders_sent": sent}
