"""Talking to the WhatsApp Business Cloud API.

Deliberately dumb, the same way the Instagram client is: this module knows how
to make the HTTP call and nothing else. The 24-hour customer service window
and the placeholder-credential convention are business rules and live in the
adapter, where delivery checks them before any call is made.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

# Pinned to a concrete version, not an alias, for the same reason the
# Instagram client pins its own: an alias can change the API's behaviour under
# a running deployment with no change on our side.
GRAPH_API_VERSION = "v21.0"
# graph.facebook.com, not graph.instagram.com. WhatsApp Cloud API tokens are
# Facebook system-user tokens, and the Instagram host refuses them.
GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# WhatsApp rejects a text body longer than this outright rather than
# truncating it. Replies here are a few sentences, so reaching it means
# something upstream went wrong -- but a patient getting the first 4096
# characters beats a patient getting nothing and a failed job.
MAX_TEXT_LENGTH = 4096


class WhatsAppSendError(Exception):
    """A send the Cloud API refused. Raised so the job fails and is retried."""


class WhatsAppClient(ABC):
    """Abstraction over "send a text message through the Cloud API", so the
    transport can be swapped for a test double without touching callers.
    """

    @abstractmethod
    async def send_text(
        self, *, access_token: str, phone_number_id: str, to: str, text: str
    ) -> None:
        """Send `text` to the WhatsApp user `to` from the business number
        `phone_number_id`. Raises WhatsAppSendError on any non-2xx response.
        """


class GraphAPIWhatsAppClient(WhatsAppClient):
    """Real implementation: POSTs to /<phone_number_id>/messages."""

    def __init__(self, *, base_url: str = GRAPH_API_BASE_URL, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def send_text(
        self, *, access_token: str, phone_number_id: str, to: str, text: str
    ) -> None:
        response = await self._http.post(
            f"/{phone_number_id}/messages",
            # A header rather than the query string the Instagram client uses:
            # the Cloud API documents bearer auth, and a token in a URL is a
            # token in every proxy log between here and Meta.
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                # preview_url off: a clinic reply that happens to contain a
                # link should not unfold into a card nobody asked for.
                "text": {"preview_url": False, "body": text[:MAX_TEXT_LENGTH]},
            },
        )
        if response.is_error:
            error_code: object = None
            error_subcode: object = None
            with suppress(ValueError):
                error = response.json().get("error", {})
                error_code = error.get("code")
                # The subcode is what separates "outside the 24-hour window"
                # (131047) from "recipient is not a WhatsApp user" and from a
                # revoked token, which all need different fixes. Scalars only:
                # Meta's error bodies can echo the request back.
                error_subcode = error.get("error_subcode")
            logger.error(
                "whatsapp_send_failed",
                extra={
                    "to": to,
                    "phone_number_id": phone_number_id,
                    "status_code": response.status_code,
                    "error_code": error_code,
                    "error_subcode": error_subcode,
                },
            )
            raise WhatsAppSendError(f"WhatsApp send failed with status {response.status_code}")


@lru_cache
def get_whatsapp_client() -> WhatsAppClient:
    return GraphAPIWhatsAppClient()
