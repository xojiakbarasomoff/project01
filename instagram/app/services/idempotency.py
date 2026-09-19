"""One-shot claims on platform events, so a redelivery cannot be answered twice.

Every messaging platform redelivers: Meta repeats a webhook whose 200 came
back too slowly or not at all, and Telegram repeats an update that was not
acknowledged. Without a claim, a redelivered message is registered again and
answered again — the patient gets the same reply twice, and the transcript
records a message they only sent once.

Redis rather than a database row because the check must be atomic against
concurrent deliveries and is pure short-lived bookkeeping: SET NX is exactly
"claim this if nobody has", in one round trip, and the key expires on its
own rather than needing a cleanup job.
"""

import hashlib
import uuid

from arq.connections import ArqRedis

# Comfortably longer than any redelivery window a platform actually uses
# (Meta retries a failed delivery for up to a few hours), short enough that
# the keyspace stays proportional to a day of traffic rather than growing
# forever.
CLAIM_TTL_SECONDS = 24 * 60 * 60


def _claim_key(tenant_id: uuid.UUID, channel_type: str, event_id: str) -> str:
    return f"event_claim:{tenant_id}:{channel_type}:{event_id}"


async def claim_event(
    pool: ArqRedis, *, tenant_id: uuid.UUID, channel_type: str, event_id: str
) -> bool:
    """Claim `event_id` for processing. True the first time, False after.

    The caller must skip the event entirely on False. Namespaced by tenant
    and channel type because a platform's ids are only unique within its own
    account — nothing stops a Telegram update id from colliding with an
    Instagram message id.
    """
    claimed = await pool.set(
        _claim_key(tenant_id, channel_type, event_id), "1", ex=CLAIM_TTL_SECONDS, nx=True
    )
    return bool(claimed)


# A reply that has already gone out over the platform's API, so a retry of the
# same job cannot send it twice.
#
# This exists because the send and the commit are not one operation and cannot
# be made into one: the Send API is a network call to Meta, the commit is a
# network call to Postgres, and there is no transaction spanning both. The
# window is real and was open -- send succeeds, commit fails, arq retries the
# job, and the patient gets the same message a second time from a database
# that has no record of the first.
#
# Claimed on the message being answered rather than on the reply text. The
# reply is regenerated on a retry and a model does not produce the same words
# twice, so keying on the reply would claim nothing; keying on what the
# patient sent is stable across every attempt at answering it.
#
# The value stored is the channel the reply went out over, because a retry
# still has to know that -- it is what decides whether the reply is written
# into the transcript.
_REPLY_TTL_SECONDS = 24 * 60 * 60


def _reply_key(conversation_id: uuid.UUID, message_text: str) -> str:
    digest = hashlib.sha256(message_text.encode("utf-8")).hexdigest()[:32]
    return f"reply_sent:{conversation_id}:{digest}"


async def claim_reply_send(
    pool: ArqRedis, *, conversation_id: uuid.UUID, message_text: str
) -> bool:
    """Claim the right to send this conversation's reply. True the first time.

    Claimed *before* the Send API call, not after: a claim taken afterwards
    would not exist if the process died between the send and the claim, which
    is the same window in a smaller form.

    The cost of claiming first is the opposite failure -- the send throws, the
    claim is already taken, and the retry stays silent. That is why
    `release_reply_claim` exists and is called whenever the send does not
    succeed. Losing a reply is recoverable (the patient writes again, an
    operator sees the conversation); sending two contradictory ones is what
    the clinic actually complained about.
    """
    claimed = await pool.set(
        _reply_key(conversation_id, message_text), "1", ex=_REPLY_TTL_SECONDS, nx=True
    )
    return bool(claimed)


async def record_reply_channel(
    pool: ArqRedis, *, conversation_id: uuid.UUID, message_text: str, channel_type: str
) -> None:
    """Remember which channel carried the reply, for a retry to read back."""
    await pool.set(
        _reply_key(conversation_id, message_text), channel_type, ex=_REPLY_TTL_SECONDS
    )


async def reply_already_sent(
    pool: ArqRedis, *, conversation_id: uuid.UUID, message_text: str
) -> str | None:
    """The channel a reply already went out over, or None if it has not."""
    value = await pool.get(_reply_key(conversation_id, message_text))
    if value is None:
        return None
    decoded = value.decode() if isinstance(value, bytes) else str(value)
    # "1" means claimed but not yet confirmed sent -- treat as not sent, so a
    # job that died mid-send is retried rather than silently dropped.
    return None if decoded == "1" else decoded


async def release_reply_claim(
    pool: ArqRedis, *, conversation_id: uuid.UUID, message_text: str
) -> None:
    """Give the claim back when the send did not happen."""
    await pool.delete(_reply_key(conversation_id, message_text))
