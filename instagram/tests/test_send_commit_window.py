"""The gap between "Meta accepted it" and "Postgres kept it".

The send is a network call to Meta. The commit is a network call to Postgres.
There is no transaction across both, so there is a window: Meta accepts the
reply, the commit then fails, arq retries the job, and the patient is sent the
same answer a second time by a database that has no record of the first.

That window was open. These tests close it and keep it closed.

The claim is taken in Redis, before the Send API call, and keyed on the
*patient's* message -- which is byte-identical on every attempt -- rather than
on the reply, which a model regenerates differently each time.
"""

import uuid

import pytest

from app.services.idempotency import (
    claim_reply_send,
    record_reply_channel,
    release_reply_claim,
    reply_already_sent,
)


@pytest.fixture
async def conversation() -> uuid.UUID:
    return uuid.uuid4()


async def test_the_first_attempt_may_send_and_the_second_may_not(
    redis_pool, conversation
) -> None:
    message = "Qabulga yozilmoqchiman"

    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text=message
    )
    assert not await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text=message
    )


async def test_a_retry_after_a_failed_commit_sends_nothing_and_knows_the_channel(
    redis_pool, conversation
) -> None:
    """The exact sequence: send succeeds, commit fails, job retries.

    The retry must redo the database work -- the lead, the state, the
    transcript -- and must not send. It also has to know which channel
    carried the reply, because that is what decides whether the reply is
    written into the transcript at all.
    """
    message = "Belim og'riyapti, qabulga yozing"

    # Attempt one: claims, sends, records the channel -- and then its commit
    # fails, which leaves Redis untouched.
    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text=message
    )
    await record_reply_channel(
        redis_pool,
        conversation_id=conversation,
        message_text=message,
        channel_type="instagram",
    )

    # Attempt two sees that the patient already has the answer.
    already = await reply_already_sent(
        redis_pool, conversation_id=conversation, message_text=message
    )
    assert already == "instagram"


async def test_a_claim_that_never_became_a_send_does_not_silence_the_retry(
    redis_pool, conversation
) -> None:
    """A job that died between claiming and sending must not leave the
    patient permanently unanswered. Claimed-but-unconfirmed reads as "not
    sent", so the retry tries again.
    """
    message = "Assalomu alaykum"

    await claim_reply_send(redis_pool, conversation_id=conversation, message_text=message)

    assert (
        await reply_already_sent(
            redis_pool, conversation_id=conversation, message_text=message
        )
        is None
    )


async def test_a_failed_send_gives_the_claim_back(redis_pool, conversation) -> None:
    """Claiming before the call buys safety against double-sending and costs
    the opposite risk: a transient Meta error would silence every retry. The
    release is what pays that back.
    """
    message = "Narxi qancha"

    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text=message
    )
    await release_reply_claim(
        redis_pool, conversation_id=conversation, message_text=message
    )

    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text=message
    )


async def test_two_different_messages_are_claimed_separately(
    redis_pool, conversation
) -> None:
    """A patient who writes twice gets two answers. The claim must not
    collapse a real second question into the first one's reply.
    """
    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text="Birinchi savol"
    )
    assert await claim_reply_send(
        redis_pool, conversation_id=conversation, message_text="Ikkinchi savol"
    )


async def test_two_patients_asking_the_same_thing_both_get_an_answer(
    redis_pool,
) -> None:
    """The key is scoped by conversation. "Salom" from two people is two
    replies, not one reply and one silence.
    """
    message = "Salom"

    assert await claim_reply_send(
        redis_pool, conversation_id=uuid.uuid4(), message_text=message
    )
    assert await claim_reply_send(
        redis_pool, conversation_id=uuid.uuid4(), message_text=message
    )
