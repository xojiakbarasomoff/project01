import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from sqlalchemy import select
from arq.connections import RedisSettings

from app.core.config import settings
from app.core.security import decrypt_credentials
from app.db.session import AsyncSessionLocal
from app.models.domain import Tenant, Channel, User, Conversation, Message, Appointment
from app.services.debounce import DebounceService
from app.services.guardrails import GuardrailService
from app.services.rag import RAGService
from app.services.llm import LLMService
from app.services.telegram import TelegramService

logger = logging.getLogger(__name__)

UZ_TZ = timezone(timedelta(hours=5))

# ── Appointment intent detection ───────────────────────────────────────────────
_BOOKING_KEYWORDS = re.compile(
    r"\b(qabul|yozib|yozing|yozilish|bormoqchiman|shifokor|konsultatsiya)\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(\+998[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    r"|\b9\d{8}\b)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\b"
    r"|\bsoat\s+\d"
    r"|\bertaga\b|\bbugun\b"
    r"|\b\d{1,2}-?\s*(avgust|sentabr|oktyabr|noyabr|dekabr|yanvar|fevral|mart|aprel|may|iyun|iyul)\b)",
    re.IGNORECASE,
)


def _extract_phone(text: str) -> str:
    """Extract first phone number from text, or empty string."""
    m = _PHONE_RE.search(text)
    return m.group(0) if m else ""


def _parse_appointment_time(text: str) -> Optional[datetime]:
    """Parse requested appointment date and time from patient text if possible (in local UTC+5 timezone)."""
    try:
        now = datetime.now(UZ_TZ)
        text_lower = text.lower()

        day_offset = 0
        if "ertaga" in text_lower:
            day_offset = 1
        elif "indinga" in text_lower:
            day_offset = 2

        target_date = now.date() + timedelta(days=day_offset)

        time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        hour, minute = None, None
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
        else:
            soat_match = re.search(r"\bsoat\s+(\d{1,2})\b", text_lower)
            if soat_match:
                hour = int(soat_match.group(1))
                minute = 0

        if hour is not None and minute is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
            return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UZ_TZ)
    except Exception as e:
        logger.warning(f"Failed to parse appointment time from text '{text}': {e}")
    return None


def _has_booking_intent(text: str) -> bool:
    """Return True when message has a phone number OR (booking keyword + time)."""
    has_phone = bool(_PHONE_RE.search(text))
    has_keyword = bool(_BOOKING_KEYWORDS.search(text))
    has_time = bool(_TIME_RE.search(text))
    return has_phone or (has_keyword and has_time)


async def process_debounce_batch(
    ctx: Dict[str, Any],
    tenant_id: int,
    user_id: int,
    external_chat_id: str,
    job_timestamp: float
):
    """
    Executes debounced message batching, guardrail evaluation, RAG semantic search,
    LLM response generation, and Telegram message delivery.
    """
    logger.info(f"🚀 [WORKER] Processing batch for tenant {tenant_id}, user {user_id}")

    # 1. Retrieve batched messages from Redis (atomic Lua transaction)
    batched_messages = await DebounceService.get_and_clear_batched_messages(
        tenant_id, user_id, job_timestamp
    )

    if batched_messages is None:
        logger.info(f"⏳ [WORKER] Batch skipped for user {user_id} (newer message debounced).")
        return True

    if not batched_messages:
        logger.info(f"ℹ️ [WORKER] No pending messages for user {user_id}.")
        return True

    combined_text = "\n".join([m["text"] for m in batched_messages if m.get("text")])
    channel_type = batched_messages[0].get("channel", "telegram")
    business_connection_id = batched_messages[-1].get("business_connection_id")

    async with AsyncSessionLocal() as session:
        # 2. Retrieve Channel credentials
        stmt = select(Channel).where(
            Channel.tenant_id == tenant_id,
            Channel.type == channel_type,
            Channel.is_active.is_(True)
        )
        res = await session.execute(stmt)
        channel = res.scalar_one_or_none()

        bot_token = ""
        if channel:
            creds = decrypt_credentials(channel.credentials)
            if isinstance(creds, dict):
                bot_token = creds.get("bot_token", "")
            elif isinstance(creds, str):
                bot_token = creds

        if not bot_token or bot_token.startswith("123456789:"):
            bot_token = settings.TELEGRAM_BOT_TOKEN

        # 3. Retrieve User
        stmt = select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        res = await session.execute(stmt)
        user_entity = res.scalar_one_or_none()

        if not user_entity:
            logger.error(f"User entity {user_id} not found in DB.")
            return False

        # 4. Retrieve or create active Conversation
        stmt = select(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
            Conversation.status != "closed"
        ).order_by(Conversation.id.desc())
        res = await session.execute(stmt)
        conversation = res.scalar_one_or_none()

        if not conversation:
            conversation = Conversation(
                tenant_id=tenant_id,
                user_id=user_id,
                status="active",
                is_bot_enabled=True
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)

        # Check Kill Switch (operator takeover)
        if not conversation.is_bot_enabled:
            logger.info(f"🛑 [WORKER] Bot disabled for conversation {conversation.id} (Operator control).")
            return True

        # 5. Send "typing..." indicator
        if channel_type == "telegram" and bot_token:
            await TelegramService.send_chat_action(
                bot_token,
                external_chat_id,
                "typing",
                business_connection_id=business_connection_id
            )

        # 6. Fetch recent conversation history BEFORE persisting the new message
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.id.desc())
            .limit(6)
        )
        res = await session.execute(stmt)
        history_msgs = list(reversed(res.scalars().all()))
        history_formatted = [
            {"role": "user" if m.sender == "patient" else "assistant", "content": m.content}
            for m in history_msgs
        ]

        # 7. Persist batched incoming message in PostgreSQL
        in_msg = Message(
            conversation_id=conversation.id,
            sender="patient",
            content=combined_text,
            channel=channel_type,
            meta={"batch_count": len(batched_messages)}
        )
        session.add(in_msg)
        await session.commit()

        # 8. Evaluate Guardrails
        guardrail_action, custom_reply = GuardrailService.check_guardrails(combined_text)

        if guardrail_action:
            logger.info(f"🛡️ [GUARDRAIL TRIGGERED] Action: {guardrail_action}")
            if guardrail_action == "OPERATOR_ESCALATION":
                conversation.is_bot_enabled = False
                conversation.status = "operator"
                await session.commit()

            sent_ok = True
            if bot_token and custom_reply:
                sent_ok = await TelegramService.send_message(
                    bot_token,
                    external_chat_id,
                    custom_reply,
                    reply_markup=TelegramService.build_booking_keyboard(),
                    business_connection_id=business_connection_id
                )

            out_msg = Message(
                conversation_id=conversation.id,
                sender="bot",
                content=custom_reply or "",
                channel=channel_type,
                meta={"guardrail_action": guardrail_action, "send_status": "sent" if sent_ok else "failed"}
            )
            session.add(out_msg)
            await session.commit()
            return True

        # 9. RAG Semantic Search
        kb_matches = await RAGService.search_knowledge_base(
            session=session,
            tenant_id=tenant_id,
            query=combined_text,
            top_k=4
        )

        # 10. LLM Response Generation
        ai_response = await LLMService.generate_response(
            user_message=combined_text,
            kb_context=kb_matches,
            conversation_history=history_formatted
        )

        # 11. Send AI Response to Telegram
        sent_ok = True
        if bot_token:
            sent_ok = await TelegramService.send_message(
                bot_token,
                external_chat_id,
                ai_response,
                reply_markup=TelegramService.build_booking_keyboard(),
                business_connection_id=business_connection_id
            )

        # 12. Save AI Response to Database
        out_msg = Message(
            conversation_id=conversation.id,
            sender="bot",
            content=ai_response,
            channel=channel_type,
            meta={"rag_matches_count": len(kb_matches), "send_status": "sent" if sent_ok else "failed"}
        )
        session.add(out_msg)

        # 13. Appointment Booking Intent Detection
        if _has_booking_intent(combined_text):
            detected_phone = _extract_phone(combined_text)
            if detected_phone and not user_entity.phone:
                user_entity.phone = detected_phone
                logger.info(f"📱 [USER] Saved phone {detected_phone} for user {user_id}")

            patient_name = user_entity.name or f"Bemor ({user_entity.external_id})"
            patient_phone = detected_phone or user_entity.phone or f"Telegram ID: {user_entity.external_id}"
            parsed_appt_time = _parse_appointment_time(combined_text)

            app_rec = Appointment(
                tenant_id=tenant_id,
                user_id=user_entity.id,
                patient_name=patient_name,
                patient_phone=patient_phone,
                doctor_name="Stomatolog Shifokor",
                appointment_time=parsed_appt_time,
                status="pending",
                notes=f"Telegram orqali so'rov: {combined_text[:200]}"
            )
            session.add(app_rec)
            logger.info(f"📅 [APPOINTMENT] Auto-created pending appointment for user {user_id}: {patient_phone}, time: {parsed_appt_time}")

        await session.commit()

        logger.info(f"✅ [WORKER] Successfully responded to user {user_id}.")
        return True


async def check_appointment_reminders(ctx: Optional[Dict[str, Any]] = None) -> int:
    """
    Cron job task that checks upcoming appointments and sends:
    1. 24-hour reminders (~22-26 hours before appointment_time)
    2. 2-hour reminders (~1-3 hours before appointment_time)
    """
    logger.info("⏰ [CRON] Checking upcoming appointment reminders...")
    now = datetime.now(UZ_TZ)
    sent_count = 0

    async with AsyncSessionLocal() as session:
        stmt = (
            select(Appointment)
            .where(
                Appointment.status != "cancelled",
                Appointment.appointment_time > now
            )
        )
        res = await session.execute(stmt)
        appointments = res.scalars().all()

        for appt in appointments:
            if not appt.appointment_time:
                continue

            diff_seconds = (appt.appointment_time - now).total_seconds()
            diff_hours = diff_seconds / 3600.0

            patient_name = appt.patient_name or "Bemor"
            doctor_name = appt.doctor_name or "Stomatolog Shifokor"
            formatted_date = appt.appointment_time.strftime("%d-%m-%Y")
            formatted_time = appt.appointment_time.strftime("%H:%M")

            target_chat_id = None
            if appt.user_id:
                u_res = await session.execute(select(User).where(User.id == appt.user_id))
                u = u_res.scalar_one_or_none()
                if u and u.external_id:
                    target_chat_id = u.external_id

            if not target_chat_id:
                if appt.patient_phone and "Telegram ID:" in appt.patient_phone:
                    target_chat_id = appt.patient_phone.replace("Telegram ID:", "").strip()

            if not target_chat_id and appt.notes and "telegram_user_id:" in appt.notes:
                m = re.search(r"telegram_user_id:(\d+)", appt.notes)
                if m:
                    target_chat_id = m.group(1)

            # ── 1. Check 24-Hour Reminder (between 22h and 26h away) ───────────
            if 22.0 <= diff_hours <= 26.0 and not appt.reminder_24h_sent:
                logger.info(f"🔔 [REMINDER] Sending 24-hour reminder to appointment #{appt.id} ({patient_name})")
                msg_text = (
                    f"🔔 <b>Qabul Eslatmasi!</b>\n\n"
                    f"Hurmatli <b>{patient_name}</b>, ertaga (<b>{formatted_date}</b>) soat <b>{formatted_time}</b> da "
                    f"<b>{doctor_name}</b> qabulida ko'rikka yozilgansiz! 😊\n\n"
                    f"Klinikamizda sizni intizorlik bilan kutamiz!"
                )
                if target_chat_id and settings.TELEGRAM_BOT_TOKEN:
                    await TelegramService.send_message(
                        bot_token=settings.TELEGRAM_BOT_TOKEN,
                        chat_id=target_chat_id,
                        text=msg_text,
                        parse_mode="HTML",
                        reply_markup=TelegramService.build_booking_keyboard()
                    )
                appt.reminder_24h_sent = True
                sent_count += 1

            # ── 2. Check 2-Hour Reminder (between 1h and 3h away) ─────────────
            elif 1.0 <= diff_hours <= 3.0 and not appt.reminder_2h_sent:
                logger.info(f"⏰ [REMINDER] Sending 2-hour reminder to appointment #{appt.id} ({patient_name})")
                msg_text = (
                    f"⏰ <b>Bugun Qabulingiz Bor!</b>\n\n"
                    f"Hurmatli <b>{patient_name}</b>, bugun soat <b>{formatted_time}</b> da "
                    f"(taxminan 2 soatdan so'ng) <b>{doctor_name}</b> qabulida ko'rikka kutilmoqdasiz! 😊\n\n"
                    f"Tashrifingizni tasdiqlaysizmi?"
                )
                if target_chat_id and settings.TELEGRAM_BOT_TOKEN:
                    await TelegramService.send_message(
                        bot_token=settings.TELEGRAM_BOT_TOKEN,
                        chat_id=target_chat_id,
                        text=msg_text,
                        parse_mode="HTML",
                        reply_markup=TelegramService.build_booking_keyboard()
                    )
                appt.reminder_2h_sent = True
                sent_count += 1

        if sent_count > 0:
            await session.commit()
            logger.info(f"✅ [CRON] Sent {sent_count} appointment reminders.")

    return sent_count


async def startup(ctx):
    logger.info("🚀 ARQ Worker started successfully")


async def shutdown(ctx):
    logger.info("🛑 ARQ Worker shutting down")


from arq import cron


class WorkerSettings:
    functions = [process_debounce_batch, check_appointment_reminders]
    cron_jobs = [cron(check_appointment_reminders, minute=set(range(0, 60, 5)))]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

