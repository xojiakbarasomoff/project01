"""Turning a platform's own account id into the tenant and channel it belongs to.

This runs BEFORE any tenant is known — it is what establishes tenant context
for an inbound event — so unlike the rest of the service layer it queries
Channel directly rather than through ChannelRepository /
TenantScopedRepository, both of which require get_current_tenant() to
already have a value.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.models.channel import Channel
from app.models.tenant import Tenant


@dataclass(frozen=True)
class ResolvedChannel:
    """Which tenant an inbound event belongs to, and over which channel.

    The channel id matters as much as the tenant: it is the account the
    patient wrote to, and therefore the only account the reply may be sent
    from. The pipeline used to carry the tenant alone and then pick a
    channel to reply through by listing the tenant's channels and taking
    whichever came back first — see app.services.delivery for what that
    cost.
    """

    tenant_id: uuid.UUID
    channel_id: uuid.UUID
    channel_type: str


async def resolve_channel(
    session: AsyncSession, *, channel_type: str, external_id: str
) -> ResolvedChannel | None:
    """The active channel with this platform account id, or None.

    None means "no channel we serve" and the event must not be processed —
    an unknown account is either a stale subscription or someone else's
    traffic, and neither has a tenant to run under.
    """
    stmt = select(Channel.tenant_id, Channel.id).where(
        Channel.type == channel_type,
        Channel.external_id == external_id,
        Channel.is_active.is_(True),
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return ResolvedChannel(tenant_id=row[0], channel_id=row[1], channel_type=channel_type)


async def resolve_instagram_channel(
    session: AsyncSession, ig_account_id: str
) -> ResolvedChannel | None:
    """resolve_channel() fixed to Instagram — what the webhook calls, so the
    channel type string lives here rather than in the route.
    """
    return await resolve_channel(
        session, channel_type=ChannelType.INSTAGRAM, external_id=ig_account_id
    )


async def bot_replies_enabled(session: AsyncSession, tenant_id: uuid.UUID) -> bool:
    """Whether this clinic's assistant is currently answering patients.

    Read from tenants.settings, the same JSONB the dashboard writes, so the
    switch on the conversations screen takes effect on the next message with
    no deploy and no restart.

    Absent means on. A clinic that has never touched the switch is a clinic
    whose bot has been answering all along, and a missing key must not read
    as "turned off" and silence it.
    """
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        return True
    return bool(tenant.settings.get("bot_replies_enabled", True))
