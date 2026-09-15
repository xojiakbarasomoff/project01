"""WhatsApp's inbound edge.

This module and app.channels.whatsapp are the only places that know what a
WhatsApp Business Cloud API webhook looks like. It does for WhatsApp exactly
what app.api.webhook does for Instagram, in the same order: authenticate the
delivery, parse it, resolve the channel, claim the message id, record the
message, and hand it to the shared services. Retrieval, guardrails, the
prompt, debounce and delivery are the ones Instagram already uses.

A Cloud API payload differs from Instagram's in shape rather than in kind:

    {"object": "whatsapp_business_account",
     "entry": [{"changes": [{"field": "messages", "value": {
         "metadata": {"phone_number_id": "..."},
         "contacts": [{"wa_id": "998901234567", "profile": {"name": "..."}}],
         "messages": [{"id": "wamid...", "from": "998901234567",
                       "type": "text", "text": {"body": "..."}}],
         "statuses":  [ ...delivered/read receipts for our own sends... ]
     }}]}]}
"""

import hashlib
import hmac
import logging

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db_session
from app.core.queue import get_arq_pool
from app.core.redaction import preview
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.services.conversation import register_inbound_message
from app.services.debounce import handle_inbound_message
from app.services.idempotency import claim_event
from app.services.profile import remember_whatsapp_contact
from app.services.tenant_resolution import (
    ResolvedChannel,
    bot_replies_enabled,
    resolve_whatsapp_channel,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook/whatsapp", tags=["WhatsApp"])


class WhatsAppText(BaseModel):
    body: str | None = None


class WhatsAppMessage(BaseModel):
    # "wamid.…" -- Meta's id for the message, and the idempotency key. The
    # Cloud API redelivers a webhook whose 200 was slow or lost, and without a
    # claim on this id the patient's message is recorded and answered twice.
    id: str
    # The sender's WhatsApp id, which is their number in international form.
    # An alias rather than a rename in __init__: "from" is a Python keyword,
    # and model_validate_json -- how the payload is actually parsed -- never
    # calls __init__, so only an alias is seen on the path that matters.
    from_: str = Field(alias="from")
    type: str
    text: WhatsAppText | None = None

    model_config = ConfigDict(populate_by_name=True)


class WhatsAppProfile(BaseModel):
    name: str | None = None


class WhatsAppContact(BaseModel):
    wa_id: str
    profile: WhatsAppProfile | None = None


class WhatsAppMetadata(BaseModel):
    phone_number_id: str


class WhatsAppValue(BaseModel):
    metadata: WhatsAppMetadata | None = None
    contacts: list[WhatsAppContact] = []
    messages: list[WhatsAppMessage] = []


class WhatsAppChange(BaseModel):
    field: str
    value: WhatsAppValue


class WhatsAppEntry(BaseModel):
    changes: list[WhatsAppChange] = []


class WhatsAppPayload(BaseModel):
    object: str
    entry: list[WhatsAppEntry] = []


def _app_secret(settings: Settings) -> str:
    # A WhatsApp number is often attached to a different Meta app from the
    # Instagram account, and each app signs with its own secret. Falling back
    # to META_APP_SECRET keeps the common single-app setup to one variable.
    return settings.whatsapp_app_secret or settings.meta_app_secret


def _verify_signature(raw_body: bytes, signature_header: str | None, app_secret: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


async def _handle_message(
    session: AsyncSession,
    pool: ArqRedis,
    channel: ResolvedChannel,
    message: WhatsAppMessage,
    contact_name: str | None,
) -> None:
    """One WhatsApp message under an already-resolved channel and tenant."""
    if message.type != "text" or message.text is None or not (message.text.body or "").strip():
        # Images, voice notes, stickers, locations: nothing for the FAQ
        # pipeline to answer yet -- the same line the Instagram edge draws
        # for attachments.
        logger.info(
            "whatsapp_non_text_skipped",
            extra={"type": message.type, "from": message.from_},
        )
        return
    text = message.text.body or ""

    if not await claim_event(
        pool,
        tenant_id=channel.tenant_id,
        channel_type=channel.channel_type,
        event_id=message.id,
    ):
        logger.info("whatsapp_duplicate_skipped", extra={"message_id": message.id})
        return

    # Length and a truncated preview only: patient content does not belong
    # in logs that may leave the host.
    logger.info(
        "whatsapp_message_received",
        extra={
            "tenant_id": str(channel.tenant_id),
            "from": message.from_,
            "message_length": len(text),
            "message_preview": preview(text),
        },
    )

    inbound = await register_inbound_message(
        session,
        channel_id=channel.channel_id,
        channel_type=channel.channel_type,
        sender_external_id=message.from_,
        text=text,
    )
    # Before every early return, for the reason the Instagram edge resolves
    # usernames there: a conversation an operator answers by hand needs a
    # name and a number on it in the dashboard just as much.
    await remember_whatsapp_contact(
        session, user_id=inbound.user_id, name=contact_name, wa_id=message.from_
    )
    await session.commit()

    if not inbound.is_bot_enabled:
        logger.info(
            "whatsapp_bot_disabled_for_conversation",
            extra={"conversation_id": str(inbound.conversation_id)},
        )
        return

    if not await bot_replies_enabled(session, channel.tenant_id):
        logger.info(
            "whatsapp_bot_disabled_for_tenant",
            extra={
                "tenant_id": str(channel.tenant_id),
                "conversation_id": str(inbound.conversation_id),
            },
        )
        return

    await handle_inbound_message(
        pool,
        tenant_id=channel.tenant_id,
        channel_id=channel.channel_id,
        conversation_id=inbound.conversation_id,
        sender_external_id=message.from_,
        message_text=text,
    )


async def _handle_payload(session: AsyncSession, pool: ArqRedis, payload: WhatsAppPayload) -> None:
    for entry in payload.entry:
        for change in entry.changes:
            # "messages" carries both inbound messages and receipts for our
            # own sends; every other field (templates, account updates) has
            # nothing for this pipeline.
            if change.field != "messages" or change.value.metadata is None:
                continue
            value = change.value
            if not value.messages:
                # Only delivery/read statuses for replies we sent.
                continue

            phone_number_id = value.metadata.phone_number_id
            channel = await resolve_whatsapp_channel(session, phone_number_id)
            if channel is None:
                logger.warning(
                    "whatsapp_unknown_phone_number_id",
                    extra={"phone_number_id": phone_number_id},
                )
                continue

            names = {
                contact.wa_id: contact.profile.name if contact.profile else None
                for contact in value.contacts
            }
            token = set_current_tenant(channel.tenant_id)
            try:
                for message in value.messages:
                    await _handle_message(
                        session, pool, channel, message, names.get(message.from_)
                    )
            finally:
                reset_current_tenant(token)


@router.get("")
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    settings: Settings = Depends(get_settings),
) -> Response:
    expected = settings.whatsapp_verify_token or settings.webhook_verify_token
    if hub_mode == "subscribe" and hub_verify_token == expected:
        return Response(content=hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Webhook verification failed")


@router.post("")
async def receive_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db_session),
    pool: ArqRedis = Depends(get_arq_pool),
) -> Response:
    raw_body = await request.body()
    if not _verify_signature(
        raw_body, request.headers.get("x-hub-signature-256"), _app_secret(settings)
    ):
        if settings.webhook_signature_enforced:
            logger.warning("whatsapp_signature_invalid body_length=%s", len(raw_body))
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature")
        logger.warning("whatsapp_signature_invalid_allowed body_length=%s", len(raw_body))

    try:
        payload = WhatsAppPayload.model_validate_json(raw_body)
    except ValueError:
        # 200, not 4xx: a payload shape this does not understand is not
        # something Meta can fix by retrying, and a retry storm helps nobody.
        logger.warning("whatsapp_payload_invalid")
        return Response(status_code=status.HTTP_200_OK)

    if payload.object != "whatsapp_business_account":
        return Response(status_code=status.HTTP_200_OK)

    await _handle_payload(session, pool, payload)
    return Response(status_code=status.HTTP_200_OK)
