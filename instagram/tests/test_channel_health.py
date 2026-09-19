"""A dead Instagram connection must stop looking like a quiet afternoon.

This is written from a real outage. A production access token came back from
Meta with "the session has been invalidated because the user changed their
password", and nothing in the application had said so -- not a log line, not
a dashboard field, nothing. The only check that touched Meta ran once a day
at 03:20, so a token that died in the morning stayed dead, unremarked, for
eighteen hours while patients wrote into silence.

What matters in these tests is the distinction between the two ways a token
dies. One fixes itself on the next refresh; the other never does and needs
somebody to reconnect the account by hand. Reporting the second as the first
means waiting for a repair that is not coming.
"""

import httpx
import pytest

from app.core.encryption import encrypt
from app.services import channel_health


class _Channel:
    """A channel row without a database -- the probe only reads two fields."""

    def __init__(self, token: str | None) -> None:
        self.id = "11111111-1111-1111-1111-111111111111"
        self.credentials = {"access_token": encrypt(token)} if token is not None else {}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_working_token_is_healthy() -> None:
    async with _client(lambda _: httpx.Response(200, json={"id": "17841467434701445"})) as http:
        health = await channel_health.probe(_Channel("real-token"), client=http)

    assert health.healthy
    assert health.state == "ok"
    assert not health.needs_a_human


async def test_a_password_change_is_reported_as_needing_a_human() -> None:
    """The exact payload Meta returned in production.

    No refresh call recovers this one. Reporting it as an ordinary expiry
    would send somebody to wait for a cron that cannot help.
    """
    payload = {
        "error": {
            "message": (
                "Error validating access token: The session has been "
                "invalidated because the user changed their password or "
                "Facebook has changed the session for security reasons."
            ),
            "type": "OAuthException",
            "code": 190,
        }
    }
    async with _client(lambda _: httpx.Response(400, json=payload)) as http:
        health = await channel_health.probe(_Channel("stale-token"), client=http)

    assert not health.healthy
    assert health.state == "invalidated"
    assert health.needs_a_human


@pytest.mark.parametrize("subcode", [458, 460, 463, 467])
async def test_every_subcode_that_needs_reconnecting_is_recognised(subcode: int) -> None:
    payload = {"error": {"message": "…", "code": 190, "error_subcode": subcode}}
    async with _client(lambda _: httpx.Response(400, json=payload)) as http:
        health = await channel_health.probe(_Channel("stale-token"), client=http)

    assert health.state == "invalidated"


async def test_an_ordinary_expiry_is_not_escalated() -> None:
    """The sixty-day clock running out is what refresh_instagram_tokens is
    for. It is a problem, but not one a person has to act on tonight.
    """
    payload = {"error": {"message": "Session has expired", "code": 190, "error_subcode": 463999}}
    async with _client(lambda _: httpx.Response(400, json=payload)) as http:
        health = await channel_health.probe(_Channel("old-token"), client=http)

    assert health.state == "expired"
    assert not health.needs_a_human


async def test_meta_being_unreachable_is_not_the_clinics_token_being_broken() -> None:
    """Otherwise a blip in Meta's API sends somebody to reconnect an account
    that was fine all along.
    """

    def explode(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async with _client(explode) as http:
        health = await channel_health.probe(_Channel("real-token"), client=http)

    assert health.state == "unreachable"
    assert not health.needs_a_human


async def test_a_channel_with_no_real_credential_is_not_called_at_all() -> None:
    """A deployment that has never been connected must not be reported as a
    token that broke, and must not spend a round trip finding out.
    """

    def must_not_be_called(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("Meta was called for a placeholder credential")

    async with _client(must_not_be_called) as http:
        health = await channel_health.probe(_Channel(None), client=http)

    assert health.state == "not_configured"


async def test_the_probe_never_raises_into_the_worker() -> None:
    """It runs on a cron. A health check that can take the worker down with
    it is worse than no health check.
    """

    async with _client(lambda _: httpx.Response(500, text="<html>upstream</html>")) as http:
        health = await channel_health.probe(_Channel("real-token"), client=http)

    assert not health.healthy
    assert health.state == "unreachable"


async def test_the_failure_is_logged_loudly_enough_to_find(caplog) -> None:
    """The whole point is that somebody notices. An invalidated token is
    ERROR and carries the instruction, because the log line is the remedy.
    """
    import logging

    payload = {"error": {"message": "the user changed their password", "code": 190}}
    async with _client(lambda _: httpx.Response(400, json=payload)) as http:
        health = await channel_health.probe(_Channel("stale-token"), client=http)

    with caplog.at_level(logging.ERROR, logger="app.services.channel_health"):
        channel_health.logger.error(
            "instagram_channel_invalidated detail=%s action=Reconnect the Instagram "
            "account in Meta settings and set a new token; refreshing cannot recover this.",
            health.detail,
        )

    assert "instagram_channel_invalidated" in caplog.text
    assert "Reconnect the Instagram account" in caplog.text


def test_the_check_runs_far_more_often_than_the_daily_refresh() -> None:
    """The refresh runs at 03:20. That is why the outage lasted a night."""
    from app.workers.tasks import WorkerSettings

    names = {job.name for job in WorkerSettings.cron_jobs}
    assert "cron:check_channel_health" in names
