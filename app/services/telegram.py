import logging
from typing import Any, Dict, Optional, Union
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot"

_IS_MOCK_TOKEN = lambda token: not token or token.startswith("123456789:")


class TelegramService:
    @staticmethod
    def get_webapp_url() -> str:
        url = getattr(settings, "WEBAPP_URL", "") or "https://almanac-trembling-tyke.ngrok-free.dev/webapp/booking.html"
        if "?" not in url:
            url = f"{url}?v=1.0.1"
        return url

    @classmethod
    def build_booking_keyboard(cls, webapp_url: Optional[str] = None) -> Dict[str, Any]:
        """Create Inline Keyboard with Booking and Location buttons."""
        url = webapp_url or cls.get_webapp_url()
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "📅 Qabulga Yozilish",
                        "web_app": {"url": url}
                    },
                    {
                        "text": "📍 Klinika Lokatsiyasi",
                        "callback_data": "get_location"
                    }
                ]
            ]
        }

    @classmethod
    def build_location_keyboard(cls, webapp_url: Optional[str] = None) -> Dict[str, Any]:
        """Create Inline Keyboard focused on Location for reminders & notifications."""
        url = webapp_url or cls.get_webapp_url()
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "📍 Klinika Lokatsiyasi (Xarita)",
                        "callback_data": "get_location"
                    },
                    {
                        "text": "📅 Qabulga Yozilish",
                        "web_app": {"url": url}
                    }
                ]
            ]
        }

    @classmethod
    def build_location_map_keyboard(
        cls,
        latitude: float = 41.311081,
        longitude: float = 69.240562,
        webapp_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create Inline Keyboard with direct Yandex Go, Google Maps, and Booking buttons."""
        url = webapp_url or cls.get_webapp_url()
        yandex_url = f"https://yandex.com/maps/?pt={longitude},{latitude}&z=17&l=map"
        google_url = f"https://www.google.com/maps?q={latitude},{longitude}"

        return {
            "inline_keyboard": [
                [
                    {
                        "text": "🚖 Yandex Go Xarita",
                        "url": yandex_url
                    },
                    {
                        "text": "📍 Google Maps Xarita",
                        "url": google_url
                    }
                ],
                [
                    {
                        "text": "📅 Qabulga Yozilish (Vizual Kalendar)",
                        "web_app": {"url": url}
                    }
                ]
            ]
        }

    @staticmethod
    async def send_message(
        bot_token: str,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
        business_connection_id: Optional[str] = None
    ) -> bool:
        """Send text message to a Telegram chat, optionally via Telegram Business connection."""
        if _IS_MOCK_TOKEN(bot_token):
            logger.info(f"[MOCK TELEGRAM OUT] Chat {chat_id}: {text[:80]}...")
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
        
        target_chat_id = chat_id
        if isinstance(chat_id, str):
            clean_id = chat_id.strip()
            if clean_id.isdigit() or (clean_id.startswith("-") and clean_id[1:].isdigit()):
                target_chat_id = int(clean_id)

        payload: Dict[str, Any] = {
            "chat_id": target_chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if business_connection_id:
            payload["business_connection_id"] = business_connection_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                if not data.get("ok"):
                    logger.error(f"Telegram API Error (payload={payload}): {data}")
                    # Fallback if Telegram Business connection fails with BUSINESS_PEER_INVALID
                    if business_connection_id and "BUSINESS_PEER_INVALID" in data.get("description", ""):
                        logger.warning(f"BUSINESS_PEER_INVALID for chat {chat_id}, retrying without business_connection_id...")
                        payload_copy = dict(payload)
                        payload_copy.pop("business_connection_id", None)
                        resp_retry = await client.post(url, json=payload_copy)
                        data_retry = resp_retry.json()
                        if data_retry.get("ok"):
                            return True
                        logger.error(f"Telegram API Retry Error: {data_retry}")
                        return False

                    return False
                return True
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {str(e)}")
                return False

    @staticmethod
    async def send_location(
        bot_token: str,
        chat_id: Union[int, str],
        latitude: float,
        longitude: float,
        business_connection_id: Optional[str] = None
    ) -> bool:
        """Send native GPS location pin to a Telegram chat."""
        if _IS_MOCK_TOKEN(bot_token):
            logger.info(f"[MOCK TELEGRAM LOCATION] Chat {chat_id}: Lat {latitude}, Lon {longitude}")
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendLocation"
        target_chat_id = chat_id
        if isinstance(chat_id, str):
            clean_id = chat_id.strip()
            if clean_id.isdigit() or (clean_id.startswith("-") and clean_id[1:].isdigit()):
                target_chat_id = int(clean_id)

        payload: Dict[str, Any] = {
            "chat_id": target_chat_id,
            "latitude": latitude,
            "longitude": longitude
        }
        if business_connection_id:
            payload["business_connection_id"] = business_connection_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                if not data.get("ok") and business_connection_id and "BUSINESS_PEER_INVALID" in data.get("description", ""):
                    payload_copy = dict(payload)
                    payload_copy.pop("business_connection_id", None)
                    resp_retry = await client.post(url, json=payload_copy)
                    data_retry = resp_retry.json()
                    return bool(data_retry.get("ok"))
                return bool(data.get("ok"))
            except Exception as e:
                logger.error(f"Failed to send Telegram location pin: {str(e)}")
                return False

    @staticmethod
    async def send_chat_action(
        bot_token: str,
        chat_id: Union[int, str],
        action: str = "typing",
        business_connection_id: Optional[str] = None
    ) -> bool:
        """Send chat action (e.g. typing status) to Telegram chat."""
        if _IS_MOCK_TOKEN(bot_token):
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendChatAction"
        
        target_chat_id = chat_id
        if isinstance(chat_id, str):
            clean_id = chat_id.strip()
            if clean_id.isdigit() or (clean_id.startswith("-") and clean_id[1:].isdigit()):
                target_chat_id = int(clean_id)

        payload: Dict[str, Any] = {"chat_id": target_chat_id, "action": action}
        if business_connection_id:
            payload["business_connection_id"] = business_connection_id

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                if not data.get("ok") and business_connection_id and "BUSINESS_PEER_INVALID" in data.get("description", ""):
                    payload_copy = dict(payload)
                    payload_copy.pop("business_connection_id", None)
                    resp_retry = await client.post(url, json=payload_copy)
                    data_retry = resp_retry.json()
                    return bool(data_retry.get("ok"))
                return bool(data.get("ok"))
            except Exception:
                return False

    @staticmethod
    async def answer_callback_query(
        bot_token: str,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False
    ) -> bool:
        """Answer callback query from inline keyboard buttons."""
        if _IS_MOCK_TOKEN(bot_token):
            return True

        url = f"{TELEGRAM_API_BASE}{bot_token}/answerCallbackQuery"
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id, "show_alert": show_alert}
        if text:
            payload["text"] = text

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                return bool(data.get("ok"))
            except Exception:
                return False

    @staticmethod
    async def set_webhook(
        bot_token: str,
        webhook_url: str,
        secret_token: Optional[str] = None
    ) -> bool:
        """Register webhook URL with Telegram Bot API, optionally with a secret token and business allowed_updates."""
        url = f"{TELEGRAM_API_BASE}{bot_token}/setWebhook"
        payload: Dict[str, Any] = {
            "url": webhook_url,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "business_connection",
                "business_message",
                "edited_business_message",
                "deleted_business_messages"
            ]
        }
        if secret_token:
            payload["secret_token"] = secret_token

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                data = response.json()
                return bool(data.get("ok"))
            except Exception as e:
                logger.error(f"Failed to set Telegram webhook: {str(e)}")
                return False
