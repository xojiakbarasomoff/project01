"""WhatsApp, from the wire in to the wire out.

No test here talks to Meta: the client's transport is an httpx MockTransport,
and the webhook's payloads are Cloud API samples written out by hand in the
shape Meta documents. What is being checked is the part that is WhatsApp's
own -- the request shape, the payload shape, and the rules a send must pass
-- since everything after the inbound edge is the pipeline Instagram already
uses and is tested there.
"""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.api.whatsapp_webhook import WhatsAppPayload, _verify_signature
from app.channels import get_adapter
from app.channels.base import CHANNEL_ACCOUNT_ID, ChannelType, DeliveryBlocked
from app.channels.whatsapp.adapter import WhatsAppAdapter
from app.channels.whatsapp.client import (
    MAX_TEXT_LENGTH,
    GraphAPIWhatsAppClient,
    WhatsAppClient,
    WhatsAppSendError,
)

PHONE_NUMBER_ID = "123456789012345"
PATIENT = "998901234567"


def _client_over(handler) -> GraphAPIWhatsAppClient:
    client = GraphAPIWhatsAppClient()
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url, transport=httpx.MockTransport(handler)
    )
    return client


# --- the client ---------------------------------------------------------------


async def test_a_reply_is_posted_to_the_business_numbers_messages_endpoint() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    await _client_over(handler).send_text(
        access_token="secret-token", phone_number_id=PHONE_NUMBER_ID, to=PATIENT, text="Salom!"
    )

    [request] = captured
    assert request.url.host == "graph.facebook.com"
    assert request.url.path.endswith(f"/{PHONE_NUMBER_ID}/messages")
    # Bearer, never the query string: a token in a URL is a token in every
    # proxy log between here and Meta.
    assert request.headers["authorization"] == "Bearer secret-token"
    assert "access_token" not in request.url.params
    assert json.loads(request.content) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": PATIENT,
        "type": "text",
        "text": {"preview_url": False, "body": "Salom!"},
    }


async def test_an_overlong_reply_is_cut_rather_than_refused() -> None:
    """Meta rejects a body past 4096 characters outright. The first 4096
    reaching the patient beats a failed job and nothing reaching them."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    await _client_over(handler).send_text(
        access_token="t", phone_number_id=PHONE_NUMBER_ID, to=PATIENT, text="x" * 5000
    )

    [request] = captured
    assert len(json.loads(request.content)["text"]["body"]) == MAX_TEXT_LENGTH


async def test_a_refused_send_raises_so_the_job_is_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Re-engagement message", "code": 131047}},
        )

    with pytest.raises(WhatsAppSendError):
        await _client_over(handler).send_text(
            access_token="t", phone_number_id=PHONE_NUMBER_ID, to=PATIENT, text="hi"
        )


# --- the adapter --------------------------------------------------------------


class _RecordingClient(WhatsAppClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def send_text(
        self, *, access_token: str, phone_number_id: str, to: str, text: str
    ) -> None:
        self.calls.append(
            {"access_token": access_token, "phone_number_id": phone_number_id, "to": to, "text": text}
        )


def test_whatsapp_is_a_registered_channel() -> None:
    assert isinstance(get_adapter("whatsapp"), WhatsAppAdapter)
    assert get_adapter("whatsapp").channel_type is ChannelType.WHATSAPP


async def test_the_adapter_sends_from_the_account_delivery_hands_it() -> None:
    client = _RecordingClient()

    await WhatsAppAdapter(client).send_text(
        credentials="token",
        recipient_external_id=PATIENT,
        text="Va alaykum assalom",
        reply_context={CHANNEL_ACCOUNT_ID: PHONE_NUMBER_ID},
    )

    assert client.calls == [
        {
            "access_token": "token",
            "phone_number_id": PHONE_NUMBER_ID,
            "to": PATIENT,
            "text": "Va alaykum assalom",
        }
    ]


@pytest.mark.parametrize("reply_context", [None, {}, {CHANNEL_ACCOUNT_ID: ""}])
async def test_a_send_with_no_business_number_is_refused_not_guessed(reply_context) -> None:
    """Sending from the wrong business number is worse than not sending."""
    client = _RecordingClient()

    with pytest.raises(ValueError, match="phone_number_id"):
        await WhatsAppAdapter(client).send_text(
            credentials="token", recipient_external_id=PATIENT, text="hi", reply_context=reply_context
        )
    assert client.calls == []


def test_a_placeholder_token_blocks_the_send() -> None:
    adapter = WhatsAppAdapter(_RecordingClient())
    assert (
        adapter.delivery_block_reason(credentials="pending", last_user_message_at=datetime.now(UTC))
        is DeliveryBlocked.NOT_CONFIGURED
    )


def test_outside_the_24_hour_window_the_send_is_blocked() -> None:
    """A free-form reply after 24 hours needs a pre-approved template, which
    this pipeline does not send. Blocked here so it costs no API call."""
    adapter = WhatsAppAdapter(_RecordingClient())
    stale = datetime.now(UTC) - timedelta(hours=25)
    fresh = datetime.now(UTC) - timedelta(hours=1)

    assert (
        adapter.delivery_block_reason(credentials="real-token", last_user_message_at=stale)
        is DeliveryBlocked.OUTSIDE_MESSAGING_WINDOW
    )
    assert adapter.delivery_block_reason(credentials="real-token", last_user_message_at=fresh) is None


# --- the webhook payload ------------------------------------------------------


def _inbound(messages: list[dict], *, statuses: list[dict] | None = None) -> bytes:
    value: dict = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "998712000393", "phone_number_id": PHONE_NUMBER_ID},
        "contacts": [{"wa_id": PATIENT, "profile": {"name": "Aziza"}}],
        "messages": messages,
    }
    if statuses is not None:
        value["statuses"] = statuses
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [{"id": "WABA_ID", "changes": [{"field": "messages", "value": value}]}],
        }
    ).encode()


def test_a_text_message_parses_with_its_sender_and_id() -> None:
    raw = _inbound(
        [
            {
                "from": PATIENT,
                "id": "wamid.HBgM",
                "timestamp": "1726000000",
                "type": "text",
                "text": {"body": "UZI qilasizlarmi?"},
            }
        ]
    )

    payload = WhatsAppPayload.model_validate_json(raw)

    [change] = payload.entry[0].changes
    assert change.value.metadata is not None
    assert change.value.metadata.phone_number_id == PHONE_NUMBER_ID
    [message] = change.value.messages
    # "from" is a Python keyword; model_validate_json never calls __init__,
    # so this only holds because the field is aliased.
    assert message.from_ == PATIENT
    assert message.id == "wamid.HBgM"
    assert message.text is not None and message.text.body == "UZI qilasizlarmi?"
    [contact] = change.value.contacts
    assert contact.profile is not None and contact.profile.name == "Aziza"


def test_a_receipt_for_our_own_reply_carries_no_messages() -> None:
    """Delivered/read statuses arrive on the same "messages" field. They must
    parse, and they must not look like a patient writing."""
    raw = json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": PHONE_NUMBER_ID},
                                "statuses": [{"id": "wamid.out", "status": "read"}],
                            },
                        }
                    ]
                }
            ],
        }
    ).encode()

    payload = WhatsAppPayload.model_validate_json(raw)

    assert payload.entry[0].changes[0].value.messages == []


def test_a_voice_note_parses_but_has_no_text() -> None:
    raw = _inbound([{"from": PATIENT, "id": "wamid.v", "type": "audio", "audio": {"id": "m1"}}])

    [message] = WhatsAppPayload.model_validate_json(raw).entry[0].changes[0].value.messages

    assert message.type == "audio"
    assert message.text is None


# --- the signature ------------------------------------------------------------


def test_only_a_body_signed_with_the_app_secret_is_accepted() -> None:
    body = _inbound([])
    good = "sha256=" + hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()

    assert _verify_signature(body, good, "app-secret") is True
    assert _verify_signature(body, good, "another-secret") is False
    assert _verify_signature(body + b" ", good, "app-secret") is False
    assert _verify_signature(body, None, "app-secret") is False
    assert _verify_signature(body, good.removeprefix("sha256="), "app-secret") is False
