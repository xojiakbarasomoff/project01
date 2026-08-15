import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

UZ_TZ = timezone(timedelta(hours=5))


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
    # Phone alone = appointment request
    # Keyword + time = appointment request
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
            Channel.is_active.is_(True)   # fix #21: use is_(True) not == True
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
        #    so the current user turn is not duplicated in the LLM prompt.
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
        #     Triggers when: phone number present OR (keyword + time expression)
        if _has_booking_intent(combined_text):
            # Extract phone from message if available, update user record
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


async def startup(ctx):
    logger.info("🚀 ARQ Worker started successfully")


async def shutdown(ctx):
    logger.info("🛑 ARQ Worker shutting down")


class WorkerSettings:
    functions = [process_debounce_batch]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
