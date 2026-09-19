"""The advisory lock, measured against a real database rather than argued about.

Two claims were made about app.services.turn.lock_conversation and neither
had been demonstrated. One of them was wrong.

**"It does not work behind PgBouncer in transaction mode."** That is false,
and it is worth writing down why, because it is a plausible-sounding thing to
believe. Transaction pooling pins a client to one server connection for the
duration of a transaction and returns it at COMMIT. `pg_advisory_xact_lock`
is taken inside a transaction and released by that same COMMIT, so its whole
lifetime sits inside the window where the connection is pinned. What breaks
under transaction pooling is the *session*-level `pg_advisory_lock`, which
outlives the transaction and so can be released onto a connection some other
client is now using. This code uses the transaction-scoped form.

(The deployment does not use PgBouncer anyway -- DATABASE_URL points at
postgres.railway.internal:5432 directly -- but the lock would be correct if
it did.)

**"Two messages from one patient are serialised."** That is the claim these
tests actually check, with two connections and real contention.
"""

import asyncio
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.services.turn import lock_conversation

# Long enough to tell waiting from not waiting, short enough that the suite
# does not crawl.
HELD_FOR_SECONDS = 0.4


@pytest.fixture
async def two_connections():
    """Two genuinely separate database connections.

    The suite's own `db_session` is one transaction that is rolled back at the
    end, so it cannot contend with itself -- and contention is the entire
    subject here.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=None)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as first, maker() as second:
        yield first, second
    await engine.dispose()


async def test_a_second_message_waits_for_the_first(two_connections) -> None:
    """The failure this prevents: two bubbles answered at once, from the same
    starting state, producing two replies that contradict each other.
    """
    first, second = two_connections
    conversation_id = uuid.uuid4()
    order: list[str] = []

    async def holder() -> None:
        await first.begin()
        await lock_conversation(first, conversation_id)
        order.append("first-locked")
        await asyncio.sleep(HELD_FOR_SECONDS)
        order.append("first-committing")
        await first.commit()

    async def waiter() -> float:
        # Started after the holder has the lock, so this is a wait and not a
        # race to acquire.
        await asyncio.sleep(HELD_FOR_SECONDS / 4)
        await second.begin()
        started = time.perf_counter()
        await lock_conversation(second, conversation_id)
        waited = time.perf_counter() - started
        order.append("second-locked")
        await second.commit()
        return waited

    _, waited = await asyncio.gather(holder(), waiter())

    assert order == ["first-locked", "first-committing", "second-locked"]
    # It really blocked, rather than returning immediately and leaving the
    # two jobs to interleave.
    assert waited > HELD_FOR_SECONDS / 2


async def test_two_different_patients_never_wait_for_each_other(two_connections) -> None:
    """The lock is per conversation. If it were not, one slow OpenAI call
    would hold up every patient in the inbox.

    Asserted against the *held* duration rather than a wall-clock constant.
    A fixed "under 100ms" bound measures the machine as much as the lock, and
    fails on a loaded CI box for reasons that have nothing to do with the
    behaviour being checked. What matters is that this acquire did not sit
    through somebody else's lock, and that comparison is machine-independent.
    """
    first, second = two_connections

    await first.begin()
    await lock_conversation(first, uuid.uuid4())

    await second.begin()
    started = time.perf_counter()
    await lock_conversation(second, uuid.uuid4())
    waited = time.perf_counter() - started

    await first.commit()
    await second.commit()

    assert waited < HELD_FOR_SECONDS / 2


async def test_the_lock_is_released_by_rollback_not_only_by_commit(two_connections) -> None:
    """A job that throws -- a provider timing out, a bad reply -- must not
    leave the patient's next message blocked behind it. The transaction-scoped
    form is what guarantees this: there is no unlock to forget.
    """
    first, second = two_connections
    conversation_id = uuid.uuid4()

    await first.begin()
    await lock_conversation(first, conversation_id)
    await first.rollback()

    await second.begin()
    started = time.perf_counter()
    await lock_conversation(second, conversation_id)
    waited = time.perf_counter() - started
    await second.commit()

    assert waited < HELD_FOR_SECONDS / 2


async def test_the_lock_is_transaction_scoped_not_session_scoped(two_connections) -> None:
    """The distinction the PgBouncer claim turned on, asserted directly.

    After COMMIT the lock must be gone from pg_locks. A session-scoped lock
    would still be held, and under transaction pooling that connection --
    still holding it -- goes back into the pool for somebody else.
    """
    first, second = two_connections
    conversation_id = uuid.uuid4()

    await first.begin()
    await lock_conversation(first, conversation_id)
    await first.commit()

    held = await second.execute(
        text(
            "SELECT count(*) FROM pg_locks "
            "WHERE locktype = 'advisory' AND objid = hashtext(:key)::bigint & 2147483647"
        ),
        {"key": str(conversation_id)},
    )
    assert held.scalar() == 0


def test_every_external_call_inside_the_lock_is_bounded() -> None:
    """The lock is only as short as the slowest call it spans.

    Three network calls happen while the per-conversation lock is held, and
    an unbounded one would hold it for as long as the provider hangs. The
    OpenAI SDK's default is ten minutes; that was the gap.
    """
    import inspect

    from app.channels.instagram.client import GraphAPIInstagramClient
    from app.rag.llm import REQUEST_TIMEOUT_SECONDS
    from app.services.sheets import TIMEOUT_SECONDS as SHEETS_TIMEOUT

    send_default = inspect.signature(
        GraphAPIInstagramClient.__init__
    ).parameters["timeout"].default

    assert REQUEST_TIMEOUT_SECONDS <= 60
    assert send_default <= 30
    assert SHEETS_TIMEOUT <= 30
