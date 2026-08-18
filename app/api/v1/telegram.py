import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decrypt_credentials
from app.db.session import get_db
from app.models.domain import Tenant, Channel, User, Conversation, Message, Appointment, Lead
from app.services.debounce import DebounceService
from app.services.rag import RAGService
from app.services.telegram import TelegramService
from app.utils.phone import extract_phone_from_text
from app.utils.telegram_helpers import get_bot_token
from app.utils.constants import LeadStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["Telegram Integration"])


def check_is_admin(tenant: Tenant, user_entity: Optional[User], sender_id: str) -> bool:
    """Checks whether a Telegram user has admin rights for the tenant."""
    if user_entity and getattr(user_entity, "is_admin", False):
        return True
    t_settings = tenant.settings if tenant and isinstance(tenant.settings, dict) else {}
    admin_ids_raw = t_settings.get("admin_telegram_ids") or ""
    if isinstance(admin_ids_raw, list):
        admin_ids = [str(x).strip() for x in admin_ids_raw if str(x).strip()]
    else:
        admin_ids = [x.strip() for x in str(admin_ids_raw).replace(";", ",").split(",") if x.strip()]

    if sender_id and sender_id in admin_ids:
        return True
    if user_entity and user_entity.external_id and str(user_entity.external_id) in admin_ids:
        return True
    return False


def _verify_webhook_secret(
    x_telegram_bot_api_secret_token: Optional[str] = Header(None)
) -> None:
    """
    Verify the Telegram webhook secret token.

    When a webhook secret is configured (TELEGRAM_WEBHOOK_SECRET in .env),
    every incoming update must carry the matching X-Telegram-Bot-Api-Secret-Token
    header.  Requests that are missing or have the wrong token are rejected with
    401 so that only Telegram can call this endpoint.

    In development (secret not set) the check is skipped.
    """
    expected = settings.TELEGRAM_WEBHOOK_SECRET
    if not expected:
        # Secret not configured — skip verification (development mode)
        return
    if x_telegram_bot_api_secret_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret token"
        )


async def send_history_user_list(
    db: AsyncSession,
    tenant_id: int,
    bot_token: str,
    chat_id: str,
    business_connection_id: Optional[str] = None
):
    """Sends an inline keyboard listing recent user conversations for selecting chat history."""
    stmt = (
        select(User)
        .where(User.tenant_id == tenant_id)
        .order_by(desc(User.id))
        .limit(10)
    )
    res = await db.execute(stmt)
    users = res.scalars().all()

    if not users:
        resp_text = "💬 Hozircha hech qanday foydalanuvchi suhbat tarixi topilmadi."
        await TelegramService.send_message(
            bot_token, chat_id, resp_text, parse_mode="HTML", business_connection_id=business_connection_id
        )
        return

    inline_keyboard = []
    for u in users:
        label = f"👤 {u.name}"
        if u.phone:
            label += f" ({u.phone})"
        elif u.external_id:
            label += f" (ID: {u.external_id})"
        inline_keyboard.append([{"text": label, "callback_data": f"hist_usr_{u.id}"}])

    reply_markup = {"inline_keyboard": inline_keyboard}
    resp_text = "💬 <b>Qaysi muloqot (bemor) suhbat tarixini ko'rmoqchisiz?</b>\n\nQuyidagi ro'yxatdan birini tanlang:"

    await TelegramService.send_message(
        bot_token, chat_id, resp_text, parse_mode="HTML", reply_markup=reply_markup, business_connection_id=business_connection_id
    )


@router.post("/webhook/{tenant_id}", dependencies=[Depends(_verify_webhook_secret)])
async def receive_telegram_webhook(
    tenant_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await _process_telegram_webhook(tenant_id, request, db)
    except Exception as e:
        logger.exception(f"Error in receive_telegram_webhook for tenant {tenant_id}: {e}")
        raise

async def _process_telegram_webhook(
    tenant_id: int,
    request: Request,
    db: AsyncSession
):
    """
    Receives incoming Telegram update webhooks, extracts tenant & user info,
    and enqueues the message into the Redis debounce pipeline.
    """
    data = await request.json()
    update_id = data.get("update_id")
    logger.info("Telegram webhook update #%s for tenant %s: %s", update_id, tenant_id, data)

    # Fix #7: Idempotency guard — skip duplicate Telegram webhook retries
    if update_id and await DebounceService.is_update_processed(tenant_id, update_id):
        return {"status": "duplicate_skipped", "update_id": update_id}

    # Handle Callback Queries (Inline Keyboard Buttons)
    callback_query = data.get("callback_query")
    if callback_query:
        cb_id = callback_query.get("id")
        cb_data = callback_query.get("data", "")
        from_user = callback_query.get("from", {})
        cb_message = callback_query.get("message", {})
        chat_id = str(cb_message.get("chat", {}).get("id") or from_user.get("id"))

        # Verify tenant & channel
        stmt = select(Tenant).where(Tenant.id == tenant_id, Tenant.status == "active")
        res = await db.execute(stmt)
        tenant = res.scalar_one_or_none()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found or inactive")

        stmt = select(Channel).where(
            Channel.tenant_id == tenant_id,
            Channel.type == "telegram",
            Channel.is_active.is_(True)
        )
        res = await db.execute(stmt)
        channel = res.scalar_one_or_none()
        bot_token = get_bot_token(channel)

        if cb_id:
            await TelegramService.answer_callback_query(bot_token, cb_id)

        if cb_data == "get_location":
            t_settings = tenant.settings if tenant and isinstance(tenant.settings, dict) else {}
            address = t_settings.get("clinic_address", "Toshkent shahri, Amir Temur shoh ko'chasi, 45-uy")
            landmark = t_settings.get("clinic_landmark", "Markaziy Universitet qarshisida")
            lat = float(t_settings.get("clinic_latitude", 41.311081))
            lng = float(t_settings.get("clinic_longitude", 69.240562))
            hours = t_settings.get("clinic_work_hours", "Har kuni 09:00 - 18:00")

            yandex_url = f"https://yandex.com/maps/?pt={lng},{lat}&z=17&l=map"
            google_url = f"https://www.google.com/maps?q={lat},{lng}"

            location_text = (
                f"📍 <b>AIMED Stomatologiya Klinikasi Manzili:</b>\n\n"
                f"🏢 <b>Manzil:</b> {address}\n"
                f"📌 <b>Mo'ljal:</b> {landmark}\n"
                f"🕒 <b>Ish vaqti:</b> {hours}\n\n"
                f"🗺 <b>Xaritalar va Navigatsiya:</b>\n"
                f"• <a href=\"{yandex_url}\">🚖 Yandex Go / Yandex Maps da ko'rish</a>\n"
                f"• <a href=\"{google_url}\">📍 Google Maps da ko'rish</a>"
            )

            if bot_token:
                await TelegramService.send_location(
                    bot_token=bot_token,
                    chat_id=chat_id,
                    latitude=lat,
                    longitude=lng
                )
                await TelegramService.send_message(
                    bot_token=bot_token,
                    chat_id=chat_id,
                    text=location_text,
                    parse_mode="HTML",
                    reply_markup=TelegramService.build_location_map_keyboard(latitude=lat, longitude=lng)
                )
            return {"status": "location_sent"}

        if cb_data.startswith("hist_usr_"):
            target_user_id = int(cb_data.replace("hist_usr_", ""))
            stmt_usr = select(User).where(User.id == target_user_id, User.tenant_id == tenant_id)
            res_usr = await db.execute(stmt_usr)
            target_user = res_usr.scalar_one_or_none()

            if not target_user:
                await TelegramService.send_message(bot_token, chat_id, "⚠️ Foydalanuvchi topilmadi.")
                return {"status": "callback_processed"}

            # Query conversations & messages for target_user
            stmt_convs = (
                select(Conversation)
                .where(Conversation.tenant_id == tenant_id, Conversation.user_id == target_user.id)
                .order_by(desc(Conversation.id))
            )
            res_convs = await db.execute(stmt_convs)
            convs = res_convs.scalars().all()
            conv_ids = [c.id for c in convs]

            if not conv_ids:
                resp_text = f"💬 <b>{target_user.name}</b> bilan suhbat tarixi mavjud emas."
            else:
                stmt_msgs = (
                    select(Message)
                    .where(Message.conversation_id.in_(conv_ids))
                    .order_by(desc(Message.id))
                    .limit(20)
                )
                res_msgs = await db.execute(stmt_msgs)
                msgs = list(reversed(res_msgs.scalars().all()))

                if not msgs:
                    resp_text = f"💬 <b>{target_user.name}</b> bilan suhbat tarixi bo'sh."
                else:
                    phone_info = f" ({target_user.phone})" if target_user.phone else ""
                    lines = [f"📋 <b>{target_user.name}</b>{phone_info} bilan suhbat tarixi:\n"]
                    for m in msgs:
                        role_icon = "👤 Bemor" if m.sender == "patient" else ("🤖 Bot" if m.sender == "bot" else "👨‍⚕️ Operator")
                        time_str = f" <i>[{m.created_at.strftime('%H:%M %d.%m')}]</i>" if m.created_at else ""
                        lines.append(f"{time_str} <b>{role_icon}:</b> {m.content}")
                    resp_text = "\n".join(lines)

            back_keyboard = {
                "inline_keyboard": [
                    [{"text": "⬅️ Barcha lichkalar ro'yxatiga qaytish", "callback_data": "hist_list"}]
                ]
            }
            await TelegramService.send_message(
                bot_token, chat_id, resp_text, parse_mode="HTML", reply_markup=back_keyboard
            )
            return {"status": "callback_processed", "user_id": target_user_id}

        elif cb_data == "hist_list":
            await send_history_user_list(db, tenant_id, bot_token, chat_id)
            return {"status": "callback_processed"}

        return {"status": "callback_ignored"}

    # Handle business_connection update (e.g. when bot is connected/disconnected or permissions updated)
    business_conn_update = data.get("business_connection")
    if business_conn_update:
        bc_id = business_conn_update.get("id")
        can_reply = business_conn_update.get("can_reply", False)
        is_enabled = business_conn_update.get("is_enabled", False)
        user_info = business_conn_update.get("user", {})
        logger.info(f"🔗 [TELEGRAM BUSINESS CONNECTION] ID: {bc_id}, User: {user_info.get('first_name')} (ID: {user_info.get('id')}), can_reply: {can_reply}, is_enabled: {is_enabled}")
        if not can_reply:
            logger.warning(f"⚠️ [TELEGRAM BUSINESS CONNECTION] 'can_reply' is FALSE for connection {bc_id}. Bot cannot reply until 'Reply to messages' permission is enabled in Telegram settings!")
        return {"status": "business_connection_updated", "can_reply": can_reply, "is_enabled": is_enabled}

    message = (
        data.get("message")
        or data.get("edited_message")
        or data.get("business_message")
        or data.get("edited_business_message")
    )
    if not message:
        return {"status": "ignored", "reason": "No message object"}

    business_connection_id = (
        message.get("business_connection_id")
        or data.get("business_connection", {}).get("id")
    )

    chat = message.get("chat", {})
    from_user = message.get("from", {})
    text = message.get("text", "")
    contact_data = message.get("contact")

    if not text and not contact_data:
        # Handle non-text messages (e.g. voice messages in Phase 2)
        if "voice" in message or "audio" in message:
            text = "[Ovozli xabar]"
        else:
            return {"status": "ignored", "reason": "Non-text message"}

    customer_id = str(chat.get("id") or from_user.get("id"))
    sender_id = str(from_user.get("id") or "")
    user_name = f"{from_user.get('first_name', '')} {from_user.get('last_name', '')}".strip() or "Bemor"

    # 1. Verify tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id, Tenant.status == "active")
    res = await db.execute(stmt)
    tenant = res.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found or inactive")

    # 2. Find active telegram channel for tenant
    stmt = select(Channel).where(
        Channel.tenant_id == tenant_id,
        Channel.type == "telegram",
        Channel.is_active.is_(True)
    )
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()

    channel_id = channel.id if channel else None

    # 3. Find or create user entity for customer
    stmt = select(User).where(User.tenant_id == tenant_id, User.external_id == customer_id)
    res = await db.execute(stmt)
    user_entity = res.scalar_one_or_none()

    if not user_entity:
        user_entity = User(
            tenant_id=tenant_id,
            channel_id=channel_id,
            external_id=customer_id,
            name=user_name
        )
        db.add(user_entity)
        await db.commit()
        await db.refresh(user_entity)

    # Handle Contact Sharing event
    if contact_data:
        phone_number = str(contact_data.get("phone_number", "")).strip()
        if phone_number and not phone_number.startswith("+"):
            phone_number = "+" + phone_number

        user_entity.phone = phone_number
        c_first = contact_data.get("first_name", "")
        c_last = contact_data.get("last_name", "")
        if c_first:
            user_entity.name = f"{c_first} {c_last}".strip()

        # Update or Create Lead
        stmt_lead = select(Lead).where(Lead.tenant_id == tenant_id, Lead.user_id == user_entity.id)
        res_lead = await db.execute(stmt_lead)
        lead_obj = res_lead.scalar_one_or_none()

        if lead_obj:
            lead_obj.phone = phone_number
            lead_obj.patient_name = user_entity.name
        else:
            lead_obj = Lead(
                tenant_id=tenant_id,
                user_id=user_entity.id,
                patient_name=user_entity.name,
                phone=phone_number,
                status="yangi",
                notes="Telegram bot orqali kontakt ulashildi"
            )
            db.add(lead_obj)

        await db.commit()
        await db.refresh(user_entity)

        chat_id = str(chat.get("id"))
        bot_token = get_bot_token(channel)

        confirm_text = (
            f"✅ <b>Rahmat, {user_entity.name}!</b>\n"
            f"Telefon raqamingiz (<b>{phone_number}</b>) muvaffaqiyatli qabul qilindi.\n\n"
            f"Endi klinika yordamchisiga o'zingizni qiziqtirgan savolingizni berishingiz yoki qabulga yozilishingiz mumkin. Qanday yordam bera olaman?"
        )
        # 1. First send confirmation and explicit remove_keyboard to close the contact request reply keyboard in Telegram app UI
        await TelegramService.send_message(
            bot_token=bot_token,
            chat_id=chat_id,
            text=confirm_text,
            parse_mode="HTML",
            reply_markup={"remove_keyboard": True},
            business_connection_id=business_connection_id
        )

        # 2. Next send inline booking action keyboard
        await TelegramService.send_message(
            bot_token=bot_token,
            chat_id=chat_id,
            text="👇 Quyidagi tugmalar orqali xizmatlar bilan tanishishingiz yoki qabulga yozilishingiz mumkin:",
            parse_mode="HTML",
            reply_markup=TelegramService.build_booking_keyboard(),
            business_connection_id=business_connection_id
        )

        if not text or text.strip() == "📱 Kontaktni ulashish":
            return {"status": "contact_received", "phone": phone_number}

    # Detect if this message was typed by the business account owner (operator) on their personal account
    is_operator_reply = bool(business_connection_id and sender_id and customer_id != sender_id)
    if is_operator_reply:
        # Find or create active conversation
        stmt_conv = (
            select(Conversation)
            .where(
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_entity.id,
                Conversation.status != "closed"
            )
            .order_by(desc(Conversation.id))
        )
        res_conv = await db.execute(stmt_conv)
        conv = res_conv.scalar_one_or_none()
        if not conv:
            conv = Conversation(tenant_id=tenant_id, user_id=user_entity.id, status="active")
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

        # Record operator message to database
        op_msg = Message(
            conversation_id=conv.id,
            sender="operator",
            content=text,
            channel="telegram"
        )
        db.add(op_msg)
        await db.commit()
        return {"status": "operator_message_recorded", "conversation_id": conv.id}

    # Helper contact keyboard for unverified users
    contact_keyboard = {
        "keyboard": [
            [{"text": "📱 Kontaktni ulashish", "request_contact": True}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

    # Check for Bot Commands (e.g. /start, /rules, /waittime, /history, /qabul)
    if text.startswith("/"):
        parts = text.strip().split()
        cmd = parts[0].lower().split("@")[0]  # strip @botusername if present
        arg = parts[1] if len(parts) > 1 else None

        bot_token = get_bot_token(channel)

        chat_id = str(chat.get("id"))
        is_admin_user = check_is_admin(tenant, user_entity, sender_id)

        if cmd in ["/start", "/help"]:
            if not is_admin_user:
                if not user_entity.phone or not str(user_entity.phone).strip():
                    start_msg = (
                        "🏥 <b>AIMED Tibbiy AI Yordamchisiga xush kelibsiz!</b>\n\n"
                        "Klinika haqida ma'lumot olish va shifokorlar qabuliga yozilish uchun "
                        "iltimos pastdagi <b>«📱 Kontaktni ulashish»</b> tugmasini bosing:"
                    )
                    await TelegramService.send_message(
                        bot_token, chat_id, start_msg, parse_mode="HTML", reply_markup=contact_keyboard, business_connection_id=business_connection_id
                    )
                    return {"status": "contact_requested", "command": cmd}
                else:
                    welcome_msg = (
                        "🏥 <b>AIMED Tibbiy AI Yordamchisiga xush kelibsiz!</b>\n\n"
                        "👇 Quyidagi tugmalar orqali xizmatlar bilan tanishishingiz yoki qabulga yozilishingiz mumkin:"
                    )
                    await TelegramService.send_message(
                        bot_token, chat_id, welcome_msg, parse_mode="HTML", reply_markup=TelegramService.build_booking_keyboard(), business_connection_id=business_connection_id
                    )
                    return {"status": "command_processed", "command": cmd}

            help_msg = (
                "🏥 <b>AIMED Tibbiy AI Yordamchi (Admin)</b>\n\n"
                "<b>Boshqaruv buyruqlari:</b>\n"
                "⏱️ /waittime [soniya] — Javob berish kutish vaqtini ko'rish yoki o'zgartirish (masalan: <code>/waittime 10</code>)\n"
                "🛡️ /rules — Tizim qat'iy qoidalari (Strict Guardrails) ko'rish va o'zgartirish\n"
                "💬 /history — Chatlar tarixini ko'rish\n"
                "📋 /qabul — Bron qilingan qabullar ro'yxati"
            )
            await TelegramService.send_message(
                bot_token, chat_id, help_msg, parse_mode="HTML", reply_markup=TelegramService.build_booking_keyboard(), business_connection_id=business_connection_id
            )
            return {"status": "command_processed", "command": cmd}

        if not is_admin_user:
            denied_msg = (
                "⚠️ <b>Ruxsat berilmadi:</b> Ushbu buyruq faqat bot administratorlari uchun mo'ljallangan.\n\n"
                "Agar siz bot admini bo'lsangiz, Telegram ID ingizni boshqaruv panelidan kiriting."
            )
            await TelegramService.send_message(
                bot_token, chat_id, denied_msg, parse_mode="HTML", business_connection_id=business_connection_id
            )
            return {"status": "access_denied", "command": cmd}

        if cmd in ["/rules", "/qoidalar", "/strict_rules", "/qoida"]:
            raw_args = text[len(parts[0]):].strip()
            if not raw_args:
                rules = await RAGService.get_strict_rules(db, tenant_id)
                rules_formatted = "\n\n".join(rules)
                resp_text = (
                    "🛡️ <b>Sun'iy Intellekt Qat'iy Qoidalari (Strict Guardrails):</b>\n\n"
                    f"{rules_formatted}\n\n"
                    "✏️ <b>Qoidalarni o'zgartirish uchun:</b>\n"
                    "<code>/rules 1. Birinchi qoida | 2. Ikkinchi qoida</code>\n"
                    "yoki qoidalarni yangi qatordan yozib yuboring:\n"
                    "<code>/rules\n"
                    "1. Birinchi qoida\n"
                    "2. Ikkinchi qoida</code>"
                )
            else:
                if "\n" in raw_args:
                    new_rules = [r.strip() for r in raw_args.split("\n") if r.strip()]
                elif "|" in raw_args:
                    new_rules = [r.strip() for r in raw_args.split("|") if r.strip()]
                else:
                    new_rules = [raw_args.strip()]

                updated_rules = await RAGService.update_strict_rules(db, tenant_id, new_rules)
                updated_formatted = "\n".join([f"• {r}" if not r.strip()[0].isdigit() else r for r in updated_rules])
                resp_text = (
                    "✅ <b>Qat'iy qoidalar muvaffaqiyatli saqlandi va faollashtirildi!</b>\n\n"
                    "🛡️ <b>Yangi qoidalar:</b>\n"
                    f"{updated_formatted}"
                )

            await TelegramService.send_message(
                bot_token, chat_id, resp_text, parse_mode="HTML", business_connection_id=business_connection_id
            )
            return {"status": "command_processed", "command": cmd}

        elif cmd in ["/waittime", "/wait", "/set_waittime"]:
            current_settings = dict(tenant.settings) if isinstance(tenant.settings, dict) else {}
            current_wait = current_settings.get("debounce_seconds", 30)

            if arg and arg.isdigit():
                new_wait = int(arg)
                if 1 <= new_wait <= 300:
                    current_settings["debounce_seconds"] = new_wait
                    tenant.settings = current_settings
                    await db.commit()
                    resp_text = f"✅ Javob berish kutish vaqti <b>{new_wait} soniya</b>ga o'zgartirildi!"
                else:
                    resp_text = "⚠️ Kutish vaqti 1 va 300 soniya orasida bo'lishi kerak."
            else:
                resp_text = (
                    f"⏱️ Hozirgi javob kutish vaqti: <b>{current_wait} soniya</b>.\n\n"
                    f"O'zgartirish uchun: <code>/waittime 10</code> (masalan: 10 soniya deb belgilash)"
                )

            await TelegramService.send_message(
                bot_token, chat_id, resp_text, parse_mode="HTML", business_connection_id=business_connection_id
            )
            return {"status": "command_processed", "command": cmd}

        elif cmd in ["/history", "/tarix", "/myhistory"]:
            await send_history_user_list(db, tenant_id, bot_token, chat_id, business_connection_id=business_connection_id)
            return {"status": "command_processed", "command": cmd}

        elif cmd in ["/qabul", "/qabullar", "/appointments"]:
            stmt_appts = (
                select(Appointment)
                .where(Appointment.tenant_id == tenant_id)
                .order_by(desc(Appointment.id))
                .limit(5)
            )
            res_appts = await db.execute(stmt_appts)
            appts = res_appts.scalars().all()

            if not appts:
                resp_text = "📋 Hozircha bron qilingan qabullar mavjud emas."
            else:
                lines = ["📋 <b>Qabullar Ro'yxati:</b>\n"]
                for idx, a in enumerate(appts, 1):
                    status_icon = "⏳" if a.status == "pending" else ("✅" if a.status == "confirmed" else "❌")
                    lines.append(
                        f"<b>{idx}. {a.patient_name}</b> ({a.patient_phone})\n"
                        f"   Holat: {status_icon} <i>{a.status}</i> | Shifokor: {a.doctor_name}\n"
                    )
                resp_text = "\n".join(lines)

            await TelegramService.send_message(
                bot_token, chat_id, resp_text, parse_mode="HTML", business_connection_id=business_connection_id
            )
            return {"status": "command_processed", "command": cmd}

    # Enforce mandatory contact for standard user text interaction
    if not user_entity.phone or not str(user_entity.phone).strip():
        typed_phone = extract_phone_from_text(text)
        if typed_phone:
            user_entity.phone = typed_phone
            stmt_lead = select(Lead).where(Lead.tenant_id == tenant_id, Lead.user_id == user_entity.id)
            res_lead = await db.execute(stmt_lead)
            lead_obj = res_lead.scalar_one_or_none()
            if lead_obj:
                lead_obj.phone = typed_phone
                lead_obj.patient_name = user_entity.name
            else:
                lead_obj = Lead(
                    tenant_id=tenant_id,
                    user_id=user_entity.id,
                    patient_name=user_entity.name,
                    phone=typed_phone,
                    status="yangi",
                    notes="Telegram bot chat orqali telefon raqam yozib qoldirildi"
                )
                db.add(lead_obj)

            await db.commit()
            await db.refresh(user_entity)

            bot_token = get_bot_token(channel)
            chat_id = str(chat.get("id"))
            confirm_text = (
                f"✅ <b>Rahmat, {user_entity.name}!</b>\n"
                f"Telefon raqamingiz (<b>{typed_phone}</b>) muvaffaqiyatli saqlandi.\n\n"
                f"Endi klinika yordamchisiga o'zingizni qiziqtirgan savolingizni berishingiz yoki qabulga yozilishingiz mumkin. Qanday yordam bera olaman?"
            )
            await TelegramService.send_message(
                bot_token=bot_token,
                chat_id=chat_id,
                text=confirm_text,
                parse_mode="HTML",
                reply_markup={"remove_keyboard": True},
                business_connection_id=business_connection_id
            )
            await TelegramService.send_message(
                bot_token=bot_token,
                chat_id=chat_id,
                text="👇 Quyidagi tugmalar orqali xizmatlar bilan tanishishingiz yoki qabulga yozilishingiz mumkin:",
                parse_mode="HTML",
                reply_markup=TelegramService.build_booking_keyboard(),
                business_connection_id=business_connection_id
            )
            # If the user only sent a phone number, finish here.
            cleaned_text = re.sub(r"[\d\+\s\-\(\)]", "", text or "").strip()
            if not cleaned_text:
                return {"status": "contact_received", "phone": typed_phone}

        else:
            bot_token = get_bot_token(channel)
            chat_id = str(chat.get("id"))
            req_msg = (
                "⚠️ <b>Muloqotni davom ettirish va savolingizga javob olish uchun kontakt ulashish majburiy!</b>\n\n"
                "Iltimos, pastdagi <b>«📱 Kontaktni ulashish»</b> tugmasini bosib telefon raqamingizni yuboring:"
            )
            await TelegramService.send_message(
                bot_token, chat_id, req_msg, parse_mode="HTML", reply_markup=contact_keyboard, business_connection_id=business_connection_id
            )
            return {"status": "contact_required"}

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
        debounce_seconds=debounce_seconds,
        business_connection_id=business_connection_id
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
    admin_token: str,  # simple shared-secret guard for the management endpoint
    db: AsyncSession = Depends(get_db)
):
    """
    Sets the Telegram webhook URL for a tenant's registered bot token.

    Requires ``admin_token`` query parameter matching TELEGRAM_WEBHOOK_SECRET
    to prevent unauthorized redirection of a tenant's bot.
    """
    # Guard the management endpoint
    expected = settings.TELEGRAM_WEBHOOK_SECRET
    if expected and admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )

    stmt = select(Channel).where(Channel.tenant_id == tenant_id, Channel.type == "telegram")
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()

    if not channel:
        raise HTTPException(status_code=404, detail="Telegram channel not found for tenant")

    creds = decrypt_credentials(channel.credentials)
    bot_token = creds.get("bot_token") if isinstance(creds, dict) else str(creds)

    if not bot_token:
        raise HTTPException(status_code=400, detail="Bot token missing in credentials")

    ok = await TelegramService.set_webhook(bot_token, webhook_url, settings.TELEGRAM_WEBHOOK_SECRET)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to set Telegram webhook")

    return {"status": "success", "webhook_url": webhook_url}
