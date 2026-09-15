"""WhatsApp as a ChannelAdapter.

Everything platform-specific about sending lives here and in the client; the
answer pipeline, debounce, conversation store and delivery never learn that
WhatsApp exists.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from app.channels.base import (
    CHANNEL_ACCOUNT_ID,
    ChannelAdapter,
    ChannelType,
    DeliveryBlocked,
)

# WhatsApp's customer service window is Meta's rule, and Meta applies it the
# same way on both of its messaging products: free-form replies are allowed
# for 24 hours after the customer last wrote. The Instagram client already
# encodes that rule and its placeholder convention, so it is reused rather
# than copied -- two copies of "24 hours" are two things to keep in step.
from app.channels.instagram.client import (
    is_placeholder_credential,
    is_within_messaging_window,
)
from app.channels.whatsapp.client import WhatsAppClient, get_whatsapp_client


class WhatsAppAdapter(ChannelAdapter):
    channel_type = ChannelType.WHATSAPP

    def __init__(self, client: WhatsAppClient | None = None) -> None:
        # Resolved lazily, as the other adapters do: registration runs at
        # import time, and building the real httpx client there would open a
        # connection pool in every process that merely imports app.channels.
        self._client = client

    def _resolve_client(self) -> WhatsAppClient:
        return self._client or get_whatsapp_client()

    async def send_text(
        self,
        *,
        credentials: str,
        recipient_external_id: str,
        text: str,
        reply_context: Mapping[str, Any] | None = None,
    ) -> None:
        phone_number_id = (reply_context or {}).get(CHANNEL_ACCOUNT_ID)
        if not isinstance(phone_number_id, str) or not phone_number_id:
            # delivery supplies this on every send; missing means a caller
            # went around delivery. Raised rather than guessed: sending from
            # the wrong business number is worse than not sending.
            raise ValueError("WhatsApp send needs the business phone_number_id to send from")
        await self._resolve_client().send_text(
            access_token=credentials,
            phone_number_id=phone_number_id,
            to=recipient_external_id,
            text=text,
        )

    def delivery_block_reason(
        self, *, credentials: str, last_user_message_at: datetime
    ) -> DeliveryBlocked | None:
        if is_placeholder_credential(credentials):
            return DeliveryBlocked.NOT_CONFIGURED
        # Outside the window a free-form message is refused by Meta and would
        # need a pre-approved template instead. Checked here so an expected
        # refusal costs no API call and fails no job.
        if not is_within_messaging_window(last_user_message_at):
            return DeliveryBlocked.OUTSIDE_MESSAGING_WINDOW
        return None


_ADAPTER = WhatsAppAdapter()


def get_whatsapp_adapter() -> WhatsAppAdapter:
    return _ADAPTER
