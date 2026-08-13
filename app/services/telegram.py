import logging
from typing import Any, Dict, Optional, Union
import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramService:
    @staticmethod
    async def send_message(
        bot_token: str,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send text message to a Telegram chat."""
        if not bot_token or bot_token.startswith("123456789:"):
            logger.info(f"[MOCK TELEGRAM OUT] Chat {chat_id}: {text}")
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                if not data.get("ok"):
                    logger.error(f"Telegram API Error: {data}")
                    return False
                return True
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {str(e)}")
                return False

    @staticmethod
    async def send_chat_action(
        bot_token: str,
        chat_id: Union[int, str],
        action: str = "typing"
    ) -> bool:
        """Send chat action (e.g. typing status) to Telegram chat."""
        if not bot_token or bot_token.startswith("123456789:"):
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendChatAction"
        payload = {
            "chat_id": chat_id,
            "action": action
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                return bool(data.get("ok"))
            except Exception:
                return False

    @staticmethod
    async def set_webhook(bot_token: str, webhook_url: str) -> bool:
        """Register webhook URL with Telegram Bot API."""
        url = f"{TELEGRAM_API_BASE}{bot_token}/setWebhook"
        payload = {"url": webhook_url}

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                return bool(data.get("ok"))
            except Exception as e:
                logger.error(f"Failed to set Telegram webhook: {str(e)}")
                return False
