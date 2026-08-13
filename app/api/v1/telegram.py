import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Tenant, Channel, User
from app.services.debounce import DebounceService
from app.services.telegram import TelegramService
from app.core.security import decrypt_credentials

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["Telegram Integration"])


@router.post("/webhook/{tenant_id}")
async def receive_telegram_webhook(
    tenant_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Receives incoming Telegram update webhooks, extracts tenant & user info,
    and enqueues the message into the Redis debounce pipeline.
    """
    data = await request.json()
    logger.info(f"Received Telegram webhook for tenant {tenant_id}: {data}")

    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"status": "ignored", "reason": "No message object"}

    chat = message.get("chat", {})
    from_user = message.get("from", {})
    text = message.get("text", "")

    if not text:
        # Handle non-text messages (e.g. voice messages in Phase 2)
        if "voice" in message or "audio" in message:
            text = "[Ovozli xabar]"
        else:
            return {"status": "ignored", "reason": "Non-text message"}

    external_id = str(from_user.get("id") or chat.get("id"))
    user_name = f"{from_user.get('first_name', '')} {from_user.get('last_name', '')}".strip() or "Bemor"

    # 1. Verify tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id, Tenant.status == "active")
    res = await db.execute(stmt)
    tenant = res.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found or inactive")

    # 2. Find or create channel
    stmt = select(Channel).where(Channel.tenant_id == tenant_id, Channel.type == "telegram")
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()

    channel_id = channel.id if channel else None

    # 3. Find or create user entity
    stmt = select(User).where(User.tenant_id == tenant_id, User.external_id == external_id)
    res = await db.execute(stmt)
    user_entity = res.scalar_one_or_none()

    if not user_entity:
        user_entity = User(
            tenant_id=tenant_id,
            channel_id=channel_id,
            external_id=external_id,
            name=user_name
        )
        db.add(user_entity)
        await db.commit()
        await db.refresh(user_entity)

    # 4. Read tenant debounce_seconds setting (default 30s)
    debounce_seconds = 30
    if tenant.settings and isinstance(tenant.settings, dict):
        debounce_seconds = int(tenant.settings.get("debounce_seconds", 30))

    # 5. Add message to Redis debounce batch pipeline
    success = await DebounceService.add_message_and_debounce(
        tenant_id=tenant_id,
        user_id=user_entity.id,
        external_chat_id=str(chat.get("id")),
        message_text=text,
        channel_type="telegram",
        debounce_seconds=debounce_seconds
    )

    return {
        "status": "enqueued" if success else "error",
        "tenant_id": tenant_id,
        "user_id": user_entity.id,
        "debounce_seconds": debounce_seconds
    }


@router.post("/set-webhook/{tenant_id}")
async def set_telegram_webhook(
    tenant_id: int,
    webhook_url: str,
    db: AsyncSession = Depends(get_db)
):
    """Sets the Telegram webhook URL for a tenant's registered bot token."""
    stmt = select(Channel).where(Channel.tenant_id == tenant_id, Channel.type == "telegram")
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()

    if not channel:
        raise HTTPException(status_code=404, detail="Telegram channel not found for tenant")

    creds = decrypt_credentials(channel.credentials)
    bot_token = creds.get("bot_token") if isinstance(creds, dict) else str(creds)

    if not bot_token:
        raise HTTPException(status_code=400, detail="Bot token missing in credentials")

    ok = await TelegramService.set_webhook(bot_token, webhook_url)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to set Telegram webhook")

    return {"status": "success", "webhook_url": webhook_url}
