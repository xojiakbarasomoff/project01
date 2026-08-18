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
            res_del = await client.post(f"{TELEGRAM_API_BASE}{token}/deleteWebhook", json={"drop_pending_updates": False})
            logger.info(f"deleteWebhook result: {res_del.json()}")
        except Exception as e:
            logger.warning(f"deleteWebhook failed: {e}")

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
                            resp = await client.post(webhook_target, json=update)
                            logger.info(f"Forward status: {resp.status_code} - {resp.text[:100]}")
                        except Exception as err:
                            logger.error(f"Failed forwarding update to web: {err}")
                            try:
                                await client.post("http://localhost:8001/api/v1/telegram/webhook/1", json=update)
                            except Exception:
                                pass
                else:
                    logger.error(f"Telegram getUpdates error: {data}")
                    await asyncio.sleep(3)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {str(e)}")
                await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(run_bot_polling())

