import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Appointment, User, Tenant
from app.services.telegram import TelegramService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webapp", tags=["Telegram WebApp Booking"])

UZ_TZ = timezone(timedelta(hours=5))


class WebappBookingSchema(BaseModel):
    tenant_id: int = 1
    doctor_name: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    patient_name: str
    patient_phone: str
    notes: Optional[str] = None
    telegram_user_id: Optional[int] = None


@router.post("/book")
async def create_webapp_booking(data: WebappBookingSchema, db: AsyncSession = Depends(get_db)):
    """Handle appointment creation directly from Telegram Mini App."""
    try:
        dt_str = f"{data.date} {data.time}"
        parsed_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=UZ_TZ)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sana yoki vaqt formati noto'g'ri"
        )

    # Link user if telegram_user_id provided or lookup by phone
    user_id = None
    if data.telegram_user_id:
        res = await db.execute(select(User).where(User.external_id == str(data.telegram_user_id)))
        usr = res.scalar_one_or_none()
        if usr:
            user_id = usr.id

    appt = Appointment(
        tenant_id=data.tenant_id,
        user_id=user_id,
        patient_name=data.patient_name,
        patient_phone=data.patient_phone,
        doctor_name=data.doctor_name,
        appointment_time=parsed_dt,
        notes=f"WebApp: {data.notes}" if data.notes else "Telegram Mini App orqali yozildi",
        status="pending"
    )
    db.add(appt)
    await db.commit()
    await db.refresh(appt)

    # If telegram bot token and telegram_user_id are present, send Telegram confirmation message
    if settings.TELEGRAM_BOT_TOKEN and data.telegram_user_id:
        confirm_msg = (
            f"✅ <b>Qabulingiz muvaffaqiyatli belgilandi!</b>\n\n"
            f"👤 <b>Bemor:</b> {data.patient_name}\n"
            f"📞 <b>Telefon:</b> {data.patient_phone}\n"
            f"👨‍⚕️ <b>Shifokor:</b> {data.doctor_name}\n"
            f"📅 <b>Vaqti:</b> {data.date} soat {data.time}\n\n"
            f"Klinikamizda sizni intizorlik bilan kutamiz! 😊"
        )
        try:
            await TelegramService.send_message(
                bot_token=settings.TELEGRAM_BOT_TOKEN,
                chat_id=data.telegram_user_id,
                text=confirm_msg,
                parse_mode="HTML"
            )
        except Exception as err:
            logger.error(f"Failed to send WebApp confirmation message: {err}")

    return {
        "status": "success",
        "appointment_id": appt.id,
        "message": "Qabul muvaffaqiyatli yaratildi"
    }
