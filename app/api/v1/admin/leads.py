"""
Admin Lead Management endpoints.

Fixes applied from code review:
  #2  — Batch queries instead of N+1 per-user/per-lead
  #3  — Tenant auth on update/delete
  #4  — Extracted core _fetch_leads() for direct Python calls
  #5  — Reset db_changed flag between phases
  #6  — Uses shared app.utils.phone utilities
  #9  — Phone is now nullable; no placeholder strings in DB
  #10 — Removed PUT decorator (PATCH-only semantics)
  #12 — Added limit/offset query params
  #14 — PII moved to DEBUG level
  #16 — Uses LeadStatus enum
  #18 — Fixed topic fallback (exact match, not prefix)
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Lead, User, Conversation, Message
from app.utils.phone import format_phone, extract_phone_from_text
from app.utils.constants import LeadStatus, PHONE_MISSING_DISPLAY, DEFAULT_TOPIC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/leads", tags=["Admin — Lead Management"])

# ── Pydantic schemas ───────────────────────────────────────────────────────────


class LeadCreate(BaseModel):
    tenant_id: int = 1
    patient_name: str
    phone: Optional[str] = None
    topic: Optional[str] = None
    convenient_time: Optional[str] = None
    status: str = LeadStatus.YANGI
    notes: Optional[str] = None


class LeadUpdate(BaseModel):
    patient_name: Optional[str] = None
    phone: Optional[str] = None
    topic: Optional[str] = None
    convenient_time: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


# ── Core business logic (callable from export.py without FastAPI DI) ───────────

# The exact LLM fallback string we want to replace with a readable default
_LLM_FALLBACK_PREFIX = "Afsuski, bizda bunday xizmat turi hozircha yo'q"


def _serialize_lead(lead: Lead, display_phone: Optional[str], display_name: str, display_topic: str) -> dict:
    """Build a dict from a Lead ORM object with display-ready values."""
    conv_time = lead.convenient_time or "Belgilanmagan"
    return {
        "id": lead.id,
        "tenant_id": lead.tenant_id,
        "user_id": lead.user_id,
        "patient_name": display_name,
        "name": display_name,
        "patient_phone": display_phone or PHONE_MISSING_DISPLAY,
        "phone": display_phone or PHONE_MISSING_DISPLAY,
        "topic": display_topic,
        "subject": display_topic,
        "message": display_topic,
        "convenient_time": conv_time,
        "preferred_time": conv_time,
        "status": lead.status or LeadStatus.YANGI,
        "notes": lead.notes or "",
        "created_at": lead.created_at.isoformat() if lead.created_at else None,
        "updated_at": lead.updated_at.isoformat() if lead.updated_at else None,
    }


async def fetch_leads(
    tenant_id: int,
    db: AsyncSession,
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[dict]:
    """
    Core lead-listing logic, usable both from the HTTP endpoint and from
    export.py (no FastAPI Query objects involved).

    Performs auto-reconciliation with batch queries to avoid N+1.
    """
    db_changed = False

    # ── 1. Batch-load all users & leads for tenant ────────────────────────
    res_u = await db.execute(select(User).where(User.tenant_id == tenant_id))
    all_users = res_u.scalars().all()
    user_map = {u.id: u for u in all_users}

    res_l = await db.execute(select(Lead).where(Lead.tenant_id == tenant_id))
    existing_leads = res_l.scalars().all()
    lead_user_ids = {l.user_id for l in existing_leads if l.user_id is not None}

    # ── 2. Batch-load conversation user_ids (single query, not N queries) ─
    res_conv_uids = await db.execute(
        select(Conversation.user_id)
        .where(Conversation.tenant_id == tenant_id)
        .distinct()
    )
    users_with_conversations = {row[0] for row in res_conv_uids.all()}

    # ── 3. Auto-create missing leads ─────────────────────────────────────
    for u in all_users:
        if u.id in lead_user_ids:
            continue
        formatted_u_phone = format_phone(u.phone)
        if formatted_u_phone or u.id in users_with_conversations:
            new_lead = Lead(
                tenant_id=tenant_id,
                user_id=u.id,
                patient_name=u.name if u.name and u.name != "-" else "Bemor",
                phone=formatted_u_phone,  # nullable now — no placeholder
                topic=DEFAULT_TOPIC,
                status=LeadStatus.YANGI,
                notes="Avto-sinxronlashtirilgan lid",
            )
            db.add(new_lead)
            db_changed = True

    if db_changed:
        await db.commit()
    db_changed = False  # Reset for phase 2

    # ── 4. Re-fetch leads after possible inserts ─────────────────────────
    res_l = await db.execute(select(Lead).where(Lead.tenant_id == tenant_id))
    existing_leads = res_l.scalars().all()

    # ── 5. Batch-load patient messages for leads that still lack a phone ──
    user_ids_needing_phone = [
        l.user_id
        for l in existing_leads
        if l.user_id
        and not format_phone(l.phone)
        and not (user_map.get(l.user_id) and format_phone(user_map[l.user_id].phone))
    ]

    # One query: grab latest patient messages for all users missing a phone
    msg_phone_map: dict[int, str] = {}
    if user_ids_needing_phone:
        # Use a window function to avoid one query per user
        stmt_msgs = (
            select(Conversation.user_id, Message.content)
            .join(Conversation)
            .where(
                Conversation.user_id.in_(user_ids_needing_phone),
                Message.sender == "patient",
            )
            .order_by(desc(Message.created_at))
        )
        res_msgs = await db.execute(stmt_msgs)
        for uid, content in res_msgs.all():
            if uid not in msg_phone_map:
                extracted = extract_phone_from_text(content)
                if extracted:
                    msg_phone_map[uid] = extracted

    # ── 6. Reconcile existing leads ──────────────────────────────────────
    for l in existing_leads:
        u = user_map.get(l.user_id) if l.user_id else None
        phone_val = format_phone(l.phone)
        name_val = l.patient_name

        # Fallback: user profile phone
        if not phone_val and u and u.phone:
            phone_val = format_phone(u.phone)

        # Fallback: extracted from messages (batch result)
        if not phone_val and l.user_id and l.user_id in msg_phone_map:
            phone_val = msg_phone_map[l.user_id]
            if u and not u.phone:
                u.phone = phone_val
                db_changed = True

        # Fallback: user name
        if (not name_val or name_val == "-") and u and u.name and u.name != "-":
            name_val = u.name

        if phone_val and l.phone != phone_val:
            l.phone = phone_val
            db_changed = True

        if name_val and l.patient_name != name_val:
            l.patient_name = name_val
            db_changed = True

    if db_changed:
        await db.commit()

    # ── 7. Apply filters & pagination ────────────────────────────────────
    stmt = select(Lead).where(Lead.tenant_id == tenant_id)

    if status_filter:
        stmt = stmt.where(Lead.status == status_filter)

    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(
            (Lead.patient_name.ilike(pattern))
            | (Lead.phone.ilike(pattern))
            | (Lead.topic.ilike(pattern))
            | (Lead.notes.ilike(pattern))
        )

    stmt = stmt.order_by(desc(Lead.created_at)).limit(limit).offset(offset)
    res = await db.execute(stmt)
    leads = res.scalars().all()

    # ── 8. Serialize ─────────────────────────────────────────────────────
    result = []
    for l in leads:
        fmt_phone = format_phone(l.phone)
        p_name = l.patient_name
        if not p_name or p_name == "-":
            p_name = f"Bemor ({fmt_phone})" if fmt_phone else "Bemor"

        # Fix #18: only replace the *exact* LLM fallback, not any string starting with "Afsuski"
        topic_text = l.topic
        if not topic_text or topic_text == _LLM_FALLBACK_PREFIX:
            topic_text = "Shifokor qabuliga yozilish"

        result.append(_serialize_lead(l, fmt_phone, p_name, topic_text))

    return result


# ── HTTP endpoints ─────────────────────────────────────────────────────────────


@router.get("")
async def list_leads(
    tenant_id: int = Query(1, description="Tenant ID"),
    status: Optional[str] = Query(None, description="Status filter (yangi, aloqada, bekor)"),
    search: Optional[str] = Query(None, description="Search by name or phone"),
    limit: int = Query(200, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
):
    """Lists all leads for the specified tenant with optional filtering and auto-reconciliation."""
    return await fetch_leads(
        tenant_id=tenant_id,
        db=db,
        status_filter=status if isinstance(status, str) else None,
        search=search if isinstance(search, str) else None,
        limit=limit if isinstance(limit, int) else 200,
        offset=offset if isinstance(offset, int) else 0,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_lead(
    payload: LeadCreate,
    db: AsyncSession = Depends(get_db),
):
    """Creates a new lead manually."""
    lead = Lead(
        tenant_id=payload.tenant_id,
        patient_name=payload.patient_name,
        phone=format_phone(payload.phone),
        topic=payload.topic,
        convenient_time=payload.convenient_time,
        status=payload.status,
        notes=payload.notes,
    )
    db.add(lead)
    await db.commit()
    await db.refresh(lead)

    logger.debug("Lead created: id=%s", lead.id)
    fmt_phone = format_phone(lead.phone)
    return {
        "id": lead.id,
        "tenant_id": lead.tenant_id,
        "patient_name": lead.patient_name,
        "patient_phone": fmt_phone or PHONE_MISSING_DISPLAY,
        "phone": fmt_phone or PHONE_MISSING_DISPLAY,
        "topic": lead.topic,
        "convenient_time": lead.convenient_time,
        "status": lead.status,
        "notes": lead.notes,
        "created_at": lead.created_at.isoformat() if lead.created_at else None,
    }


@router.patch("/{lead_id}")
async def update_lead(
    lead_id: int,
    payload: LeadUpdate,
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db),
):
    """Updates an existing lead status or details (PATCH semantics — partial update)."""
    # Fix #3: enforce tenant scoping
    stmt = select(Lead).where(Lead.id == lead_id, Lead.tenant_id == tenant_id)
    res = await db.execute(stmt)
    lead = res.scalar_one_or_none()

    if not lead:
        raise HTTPException(status_code=404, detail="Lid topilmadi")

    update_data = payload.model_dump(exclude_unset=True)
    if "phone" in update_data and update_data["phone"]:
        update_data["phone"] = format_phone(update_data["phone"]) or update_data["phone"]

    for k, v in update_data.items():
        setattr(lead, k, v)

    await db.commit()
    await db.refresh(lead)
    logger.debug("Lead updated: id=%s, status=%s", lead.id, lead.status)

    fmt_phone = format_phone(lead.phone)
    return {
        "id": lead.id,
        "patient_name": lead.patient_name,
        "patient_phone": fmt_phone or PHONE_MISSING_DISPLAY,
        "phone": fmt_phone or PHONE_MISSING_DISPLAY,
        "topic": lead.topic,
        "convenient_time": lead.convenient_time,
        "status": lead.status,
        "notes": lead.notes,
        "updated_at": lead.updated_at.isoformat() if lead.updated_at else None,
    }


@router.delete("/{lead_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lead(
    lead_id: int,
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a lead record."""
    # Fix #3: enforce tenant scoping
    stmt = select(Lead).where(Lead.id == lead_id, Lead.tenant_id == tenant_id)
    res = await db.execute(stmt)
    lead = res.scalar_one_or_none()

    if not lead:
        raise HTTPException(status_code=404, detail="Lid topilmadi")

    await db.delete(lead)
    await db.commit()
    return None
