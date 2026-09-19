"""Is the clinic's Instagram connection actually alive?

A dead Instagram token looks exactly like a quiet afternoon. Nothing raises,
nothing retries, and the only sign is an absence: no replies going out, and --
once Meta drops the connection -- no webhooks coming in either. The clinic
notices when a patient telephones to ask why nobody answered.

That is how this was found. A production ACCESS_TOKEN came back from Meta
with "the session has been invalidated because the user changed their
password", and nothing anywhere in the application had said so. The only
existing check was `refresh_instagram_tokens`, which runs once a day at 03:20:
a token that dies at 09:00 stays dead, silently, for eighteen hours.

So this asks Meta one read-only question -- who does this token belong to? --
and says loudly when the answer is an error. It changes nothing, books
nothing, and sends nothing. Its entire output is a log line, which is the
point: an outage that is visible in the first ten minutes is an outage
somebody can fix before the clinic loses a day of messages.

Distinguishing the two failures matters, because the remedies are different
and one of them needs a human:

  * **expired** -- the sixty-day clock ran out. `refresh_instagram_tokens`
    handles this on its own, provided it runs before the expiry.
  * **invalidated** -- the password changed, or Meta cut the session for
    security. No refresh can recover it. Somebody has to reconnect the
    account in Meta's settings and put the new token in. Logged at ERROR with
    that instruction, because it is the only case where waiting does not help.
"""

import logging
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.instagram.client import GRAPH_API_BASE_URL, is_placeholder_credential
from app.core.encryption import DecryptionError, decrypt
from app.models.channel import Channel

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 15.0

# Meta's code for every token problem. The subcode is what separates "this
# expired on its own" from "somebody changed the password".
_OAUTH_ERROR = 190
# Subcodes that mean a human has to reconnect the account: no refresh call
# will bring these back.
_NEEDS_A_HUMAN = frozenset(
    {
        # The user removed the app.
        458,
        # The password changed, which invalidates every session with it.
        460,
        # Meta invalidated the token for a security reason of its own.
        463,
        # The user must re-authenticate.
        467,
    }
)


@dataclass(frozen=True)
class ChannelHealth:
    """What one probe found."""

    channel_id: str
    healthy: bool
    # "ok", "not_configured", "expired", "invalidated", "unreachable".
    state: str
    detail: str = ""

    @property
    def needs_a_human(self) -> bool:
        return self.state == "invalidated"


async def probe(
    channel: Channel, *, client: httpx.AsyncClient | None = None
) -> ChannelHealth:
    """Ask Meta whether this channel's token still works.

    `/me` is the cheapest question there is and needs no extra permission
    beyond the one the channel already has. A healthy token answers with an
    id; a dead one answers with an error whose subcode says which kind of
    dead it is.
    """
    channel_id = str(channel.id)
    try:
        token = decrypt(channel.credentials.get("access_token", ""))
    except (DecryptionError, AttributeError):
        return ChannelHealth(channel_id, False, "not_configured", "credentials unreadable")
    if not token or is_placeholder_credential(token):
        return ChannelHealth(channel_id, False, "not_configured", "placeholder credential")

    owned = client is None
    http = client or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        response = await http.get(
            f"{GRAPH_API_BASE_URL}/me",
            params={"fields": "id", "access_token": token},
        )
    except httpx.HTTPError as exc:
        # Meta being unreachable is not the clinic's token being broken, and
        # saying so would send somebody to reconnect an account that is fine.
        return ChannelHealth(channel_id, False, "unreachable", type(exc).__name__)
    finally:
        if owned:
            await http.aclose()

    if not response.is_error:
        return ChannelHealth(channel_id, True, "ok")

    code: object = None
    subcode: object = None
    message = ""
    try:
        error = response.json().get("error", {})
        code, subcode = error.get("code"), error.get("error_subcode")
        # Meta's message names the cause in plain English and carries no
        # credential; the rest of the payload can echo the token back, so
        # only this field is kept.
        message = str(error.get("message", ""))[:200]
    except ValueError:
        pass

    if code == _OAUTH_ERROR and (subcode in _NEEDS_A_HUMAN or "password" in message.lower()):
        return ChannelHealth(channel_id, False, "invalidated", message)
    if code == _OAUTH_ERROR:
        return ChannelHealth(channel_id, False, "expired", message)
    return ChannelHealth(channel_id, False, "unreachable", f"status {response.status_code}")


async def check_instagram_channels(
    session: AsyncSession, *, client: httpx.AsyncClient | None = None
) -> list[ChannelHealth]:
    """Probe every Instagram channel and log what is wrong with each.

    Never raises. It runs on a cron beside the webhook watch, and a health
    check that can take the worker down with it is worse than no health check.
    """
    # Imported here rather than at module scope: token_refresh imports this
    # module's siblings, and the cycle is not worth a shared helper.
    from app.services.token_refresh import _instagram_channels

    results: list[ChannelHealth] = []
    for channel in await _instagram_channels(session):
        try:
            health = await probe(channel, client=client)
        except Exception:  # noqa: BLE001 - a broken probe must not stop the worker
            logger.exception("instagram_health_probe_failed", extra={"channel_id": str(channel.id)})
            continue
        results.append(health)

        if health.healthy:
            logger.info("instagram_channel_healthy", extra={"channel_id": health.channel_id})
        elif health.needs_a_human:
            logger.error(
                "instagram_channel_invalidated "
                "detail=%s "
                "action=Reconnect the Instagram account in Meta settings and set a new "
                "token; refreshing cannot recover this.",
                health.detail,
                extra={"channel_id": health.channel_id},
            )
        elif health.state == "unreachable":
            logger.warning(
                "instagram_channel_unreachable",
                extra={"channel_id": health.channel_id, "detail": health.detail},
            )
        else:
            logger.error(
                "instagram_channel_unhealthy",
                extra={
                    "channel_id": health.channel_id,
                    "state": health.state,
                    "detail": health.detail,
                },
            )
    return results
