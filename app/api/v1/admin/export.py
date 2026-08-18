"""
Admin CSV export endpoints.

Fix #4: Calls the plain ``fetch_leads()`` function directly instead of
the FastAPI-decorated ``list_leads()`` — no more Query descriptor objects
leaking into business logic.
"""

import csv
import io
import logging
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Appointment
from app.api.v1.admin.leads import fetch_leads
from app.utils.constants import PHONE_MISSING_DISPLAY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/export", tags=["Admin — Export"])


@router.get("/appointments")
async def export_appointments_csv(
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Exports all clinic appointments as a UTF-8 BOM CSV file,
    fully compatible with MS Excel.
    """
    stmt = (
        select(Appointment)
        .where(Appointment.tenant_id == tenant_id)
        .order_by(desc(Appointment.created_at))
    )
    res = await db.execute(stmt)
    appointments = res.scalars().all()

    output = io.StringIO()
    # Write UTF-8 BOM so Excel displays Cyrillic / Uzbek characters properly
    output.write("\ufeff")

    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Bemor Ismi",
        "Telefon Raqami",
        "Shifokor",
        "Qabul Vaqti",
        "Holati",
        "Izoh",
        "Yaratilgan Vaqt",
    ])

    for appt in appointments:
        appt_time_str = (
            appt.appointment_time.strftime("%d-%m-%Y %H:%M")
            if appt.appointment_time
            else "Belgilanmagan"
        )
        created_str = appt.created_at.strftime("%d-%m-%Y %H:%M") if appt.created_at else ""

        writer.writerow([
            appt.id,
            appt.patient_name or "Noma'lum",
            appt.patient_phone or "",
            appt.doctor_name or "Stomatolog",
            appt_time_str,
            appt.status.upper(),
            appt.notes or "",
            created_str,
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=Qabullar_Roixati_AIMED.csv"},
    )


@router.get("/leads")
async def export_leads_csv(
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Exports all clinic leads as a UTF-8 BOM CSV file,
    fully compatible with MS Excel.
    """
    # Fix #4: call the plain function, not the FastAPI endpoint
    leads = await fetch_leads(tenant_id=tenant_id, db=db)

    output = io.StringIO()
    output.write("\ufeff")  # UTF-8 BOM for Excel

    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Bemor Ismi",
        "Telefon Raqami",
        "Murojaat/Mavzu",
        "Qulay Vaqt",
        "Holati",
        "Izoh",
        "Yaratilgan Vaqt",
    ])

    for l in leads:
        writer.writerow([
            l.get("id"),
            l.get("patient_name") or "Bemor",
            l.get("phone") or PHONE_MISSING_DISPLAY,
            l.get("topic") or "",
            l.get("convenient_time") or "",
            (l.get("status") or "yangi").upper(),
            l.get("notes") or "",
            l.get("created_at") or "",
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=Lidlar_Roixati_AIMED.csv"},
    )
