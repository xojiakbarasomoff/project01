"""
Telegram-specific helper utilities shared by the webhook handler and worker.
"""

from typing import Optional

from app.core.config import settings
from app.core.security import decrypt_credentials


def get_bot_token(channel) -> str:
    """
    Extract the bot token from a Channel ORM object.

    Falls back to the global ``TELEGRAM_BOT_TOKEN`` setting when the
    channel is None or its credentials are missing / placeholder.
    """
    env_token = (settings.TELEGRAM_BOT_TOKEN or "").strip().strip('"').strip("'")
    if not channel:
        return env_token

    creds = decrypt_credentials(channel.credentials)

    if isinstance(creds, dict):
        token = creds.get("bot_token", "")
    elif isinstance(creds, str):
        token = creds
    else:
        token = ""

    token = str(token).strip().strip('"').strip("'")

    # Guard against placeholder tokens left over from dev fixtures
    if not token or token.startswith("123456789:"):
        return env_token

    return token

