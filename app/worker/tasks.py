import asyncio
import logging
from typing import Dict, Any
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

    # 1. Retrieve batched messages from Redis
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

    async with AsyncSessionLocal() as session:
        # 2. Retrieve Tenant & Channel credentials
        stmt = select(Channel).where(Channel.tenant_id == tenant_id, Channel.type == channel_type, Channel.is_active == True)
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

        # 3. Retrieve User & Conversation status
        stmt = select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        res = await session.execute(stmt)
        user_entity = res.scalar_one_or_none()

        if not user_entity:
            logger.error(f"User entity {user_id} not found in DB.")
            return False

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

        # Check Kill Switch status (if operator has taken over chat)
        if not conversation.is_bot_enabled:
            logger.info(f"🛑 [WORKER] Bot is disabled for conversation {conversation.id} (Operator control).")
            return True

        # 4. Indicate "typing..." to Telegram user
        if channel_type == "telegram" and bot_token:
            await TelegramService.send_chat_action(bot_token, external_chat_id, "typing")

        # 5. Persist batched incoming message in PostgreSQL
        in_msg = Message(
            conversation_id=conversation.id,
            sender="patient",
            content=combined_text,
            channel=channel_type,
            meta={"batch_count": len(batched_messages)}
        )
        session.add(in_msg)
        await session.commit()

        # 6. Evaluate Guardrails
        guardrail_action, custom_reply = GuardrailService.check_guardrails(combined_text)

        if guardrail_action:
            logger.info(f"🛡️ [GUARDRAIL TRIGGERED] Action: {guardrail_action}")
            if guardrail_action == "OPERATOR_ESCALATION":
                conversation.is_bot_enabled = False
                conversation.status = "operator"
                await session.commit()

            # Send guardrail response
            if bot_token:
                await TelegramService.send_message(bot_token, external_chat_id, custom_reply)

            # Persist response in DB
            out_msg = Message(
                conversation_id=conversation.id,
                sender="bot",
                content=custom_reply,
                channel=channel_type,
                meta={"guardrail_action": guardrail_action}
            )
            session.add(out_msg)
            await session.commit()
            return True

        # 7. RAG Semantic Search
        kb_matches = await RAGService.search_knowledge_base(
            session=session,
            tenant_id=tenant_id,
            query=combined_text,
            top_k=4
        )

        # 8. Fetch recent conversation history
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

        # 9. LLM Response Generation
        ai_response = await LLMService.generate_response(
            user_message=combined_text,
            kb_context=kb_matches,
            conversation_history=history_formatted
        )

        # 10. Send AI Response to Telegram
        if bot_token:
            await TelegramService.send_message(bot_token, external_chat_id, ai_response)

        # 11. Save AI Response to Database
        out_msg = Message(
            conversation_id=conversation.id,
            sender="bot",
            content=ai_response,
            channel=channel_type,
            meta={"rag_matches_count": len(kb_matches)}
        )
        session.add(out_msg)

        # 12. Check for Appointment Booking Intent & Auto-create Appointment Record
        booking_keywords = ["qabul", "yozib", "yozing", "ertaga", "bugun", "soat", "bormoqchiman", "yozilish"]
        if any(kw in combined_text.lower() for kw in booking_keywords):
            # Create a pending appointment for the operator dashboard
            patient_name = user_entity.name or f"Bemor ({user_entity.external_id})"
            patient_phone = user_entity.phone or f"Telegram ID: {user_entity.external_id}"
            
            app_rec = Appointment(
                tenant_id=tenant_id,
                user_id=user_entity.id,
                patient_name=patient_name,
                patient_phone=patient_phone,
                doctor_name="Stomatolog Shifokor",
                status="pending",
                notes=f"Telegram orqali so'rov: {combined_text[:120]}"
            )
            session.add(app_rec)
            logger.info(f"📅 [APPOINTMENT] Auto-created pending appointment for user {user_id}")

        await session.commit()

        logger.info(f"✅ [WORKER] Successfully responded to user {user_id}.")
        return True


async def startup(ctx):
    logger.info("🚀 ARQ Worker started successfully")


async def shutdown(ctx):
    logger.info("🛑 ARQ Worker shutting down")


class WorkerSettings:
    functions = [process_debounce_batch]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
