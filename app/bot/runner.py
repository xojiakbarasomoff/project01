import asyncio
import logging
import httpx
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_bot_runner")

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


async def run_bot_polling():
    token = settings.TELEGRAM_BOT_TOKEN
    if not token or token.startswith("123456789:"):
        logger.warning(
            "\n======================================================\n"
            "⚠️ TELEGRAM_BOT_TOKEN is currently set to dummy placeholder in .env!\n"
            "To connect a real live Telegram bot:\n"
            "1. Create a bot using @BotFather on Telegram\n"
            "2. Set TELEGRAM_BOT_TOKEN=your_real_bot_token in .env\n"
            "3. Restart containers: docker compose restart bot\n"
            "======================================================\n"
        )
        while True:
            await asyncio.sleep(3600)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Delete webhook first so polling works
        try:
            await client.post(f"{TELEGRAM_API_BASE}{token}/deleteWebhook")
        except Exception:
            pass

        offset = 0
        while True:
            try:
                url = f"{TELEGRAM_API_BASE}{token}/getUpdates"
                response = await client.get(url, params={"offset": offset, "timeout": 20})
                data = response.json()

                if data.get("ok"):
                    updates = data.get("result", [])
                    for update in updates:
                        offset = update["update_id"] + 1
                        logger.info(f"📥 Received live Telegram update #{update['update_id']}")

                        # Forward to local webhook endpoint
                        webhook_target = "http://web:8000/api/v1/telegram/webhook/1"
                        try:
                            await client.post(webhook_target, json=update)
                        except Exception as err:
                            # Fallback local URL if running outside docker
                            await client.post("http://localhost:8001/api/v1/telegram/webhook/1", json=update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {str(e)}")
                await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(run_bot_polling())
