import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Conversation, Message, User, Channel
from app.services.telegram import TelegramService
from app.core.security import decrypt_credentials

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/conversations", tags=["Admin — Live Conversations & Operator Control"])


@router.get("")
async def list_conversations(
    tenant_id: int = Query(1, description="Tenant ID"),
    db: AsyncSession = Depends(get_db)
):
    """Lists all active and ongoing patient conversations."""
    stmt = (
        select(Conversation, User)
        .join(User, Conversation.user_id == User.id)
        .where(Conversation.tenant_id == tenant_id)
        .order_by(desc(Conversation.updated_at))
    )
    res = await db.execute(stmt)
    rows = res.all()

    result = []
    for conv, user_entity in rows:
        # Fetch last message
        stmt_last = (
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(desc(Message.id))
            .limit(1)
        )
        res_last = await db.execute(stmt_last)
        last_msg = res_last.scalar_one_or_none()

        result.append({
            "id": conv.id,
            "tenant_id": conv.tenant_id,
            "user_id": conv.user_id,
            "patient_name": user_entity.name or "Bemor",
            "patient_phone": user_entity.phone or "-",
            "external_id": user_entity.external_id,
            "status": conv.status,
            "is_bot_enabled": conv.is_bot_enabled,
            "last_message": last_msg.content if last_msg else None,
            "last_message_sender": last_msg.sender if last_msg else None,
            "updated_at": conv.updated_at
        })

    return result


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Retrieves full message history for a conversation."""
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    res = await db.execute(stmt)
    messages = res.scalars().all()
    return messages


@router.post("/{conversation_id}/toggle-bot")
async def toggle_bot_switch(
    conversation_id: int,
    enable: Optional[bool] = None,
    db: AsyncSession = Depends(get_db)
):
    """Toggles AI bot control on/off (Kill Switch / Operator Takeover)."""
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    res = await db.execute(stmt)
    conv = res.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if enable is not None:
        conv.is_bot_enabled = enable
    else:
        conv.is_bot_enabled = not conv.is_bot_enabled

    conv.status = "active" if conv.is_bot_enabled else "operator"
    await db.commit()
    await db.refresh(conv)

    return {
        "status": "success",
        "conversation_id": conv.id,
        "is_bot_enabled": conv.is_bot_enabled,
        "conversation_status": conv.status
    }


@router.post("/{conversation_id}/send-message")
async def operator_send_message(
    conversation_id: int,
    text: str,
    db: AsyncSession = Depends(get_db)
):
    """Allows human operator to send a direct message to the patient via Telegram."""
    stmt = (
        select(Conversation, User)
        .join(User, Conversation.user_id == User.id)
        .where(Conversation.id == conversation_id)
    )
    res = await db.execute(stmt)
    row = res.first()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv, user_entity = row

    # Fetch channel token
    stmt_ch = select(Channel).where(Channel.tenant_id == conv.tenant_id, Channel.type == "telegram")
    res_ch = await db.execute(stmt_ch)
    channel = res_ch.scalar_one_or_none()

    if not channel:
        raise HTTPException(status_code=400, detail="Telegram channel configuration missing")

    creds = decrypt_credentials(channel.credentials)
    bot_token = creds.get("bot_token") if isinstance(creds, dict) else str(creds)

    # Disable bot automatically when operator sends message
    conv.is_bot_enabled = False
    conv.status = "operator"

    # Send message via Telegram API
    await TelegramService.send_message(bot_token, user_entity.external_id, f"👤 <b>Operator:</b> {text}")

    # Persist operator message in DB
    msg = Message(
        conversation_id=conv.id,
        sender="operator",
        content=text,
        channel="telegram",
        meta={"manual_override": True}
    )
    db.add(msg)
    await db.commit()

    return {"status": "sent", "message_id": msg.id}
