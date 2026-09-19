"""One inbound message, from what the clinic remembers to what it says back.

This is the part that was missing. Before it, answering a message meant:
fetch the last ten turns, hand them to the model with some retrieved FAQs,
send whatever came out. There was nowhere for the patient's name to be
written down, nothing that knew a booking was half-collected, nothing that
stopped two jobs answering the same person at once, and nothing that checked
whether a sentence claiming an appointment was true.

The order below is the design, and each step exists because of a specific
thing that went wrong:

  1. **Lock the conversation.** Two bubbles arriving together used to produce
     two replies that contradicted each other. A transaction-scoped advisory
     lock makes the second job wait for the first, so it sees the state the
     first one wrote rather than the state both started from.
  2. **Read what the message contains** -- name, telephone, language -- and
     write it to the patient's row before anything decides what to say. This
     is why a name is never asked for twice.
  3. **Classify the intent** against the state, so "ha" and "12" mean what
     the open question makes them mean.
  4. **Handle the flow in code** where the flow is deterministic: a day the
     clinic is closed is refused when the patient names it, not after they
     have agreed to everything.
  5. **Only then ask the model**, and only for language.
  6. **Check what it said.** A reply claiming an appointment that no row
     backs is replaced, not sent.

The model is never the authority on a fact. It phrases; the database decides.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_state import ConversationState, FlowStatus
from app.repositories.conversation_state import ConversationStateRepository
from app.repositories.user import UserRepository
from app.services import booking_request, patient_profile
from app.services import when as when_service
from app.services.intent import BOOKING_INTENTS, Intent, classify

logger = logging.getLogger(__name__)


async def lock_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    """Serialise everything this transaction does for one conversation.

    Transaction-scoped (`pg_advisory_xact_lock`), so it is released by commit
    or rollback and cannot be leaked by an exception -- there is no unlock to
    forget. Keyed on the conversation rather than the patient because the
    conversation is what has a state row.

    Two patients never wait for each other; two messages from one patient
    always do, which is exactly the contention that produced contradictory
    replies.

    **How far it reaches.** The lock lives until the transaction ends, and
    app.workers.tasks.process_inbound_message now has exactly one commit, at
    the very end. So one job holds it across all of this:

        lock -> read profile -> classify -> write state
             -> [generate_answer: OpenAI embeddings + completion]
             -> [settle_booking: flush, no commit]
             -> [send_reply: the platform's Send API]
             -> [mirror_lead / mirror_appointment: Google Sheets]
             -> record_outbound_message -> COMMIT (lock released)

    That was not true before. An intermediate `commit()` sat in the middle of
    that sequence and released the lock early, leaving the send and the
    transcript write unprotected -- the exact window in which a second
    message could overtake the first.

    **Three external calls happen inside the lock**, and that is deliberate
    rather than overlooked. The reply the patient receives has to be the one
    built from the state this job wrote, so the send cannot be moved outside.
    The costs are real and bounded: the lock is per conversation, so it
    delays nobody else; a patient's bubbles are already batched by the
    debounce window before a job is queued; and the job's own retry/backoff
    (_RETRY_BACKOFF_SECONDS) bounds how long a stuck provider can hold it.
    What it does mean is that a slow OpenAI call makes a second message from
    the *same* patient wait, which is the intended trade: late is recoverable,
    contradictory is not.
    """
    await session.execute(
        sql_text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": str(conversation_id)},
    )


@dataclass(frozen=True)
class Outcome:
    """What this turn decided, for the caller to send and record."""

    reply: str
    # True when code produced the whole reply and the model was not asked.
    # The flow questions are fixed wording, and a model given the chance to
    # rephrase them eventually asks two things at once.
    handled_in_code: bool = False
    # Set when the patient gave a second, different telephone number.
    conflicting_phone: str | None = None


# Sentences that claim an appointment exists. Matched against what the model
# produced, because a prompt instruction not to say them is a rule waiting
# for a wording it does not cover -- and this one has to hold every time.
_CLAIMS_A_BOOKING = re.compile(
    r"yozib\s*qo'?y|yozib\s*qo’|yozildingiz|yozib\s*oldim|band\s*qildim"
    r"|qabulingiz\s+tasdiq|tasdiqlandi|navbatingiz\s+bor"
    r"|ёзиб\s*қўй|ёзилдингиз|тасдиқланди"
    r"|записал\s+вас|вы\s+записан|запись\s+подтвержден"
    r"|\bbooked\b|\byou are booked\b|\bconfirmed\b",
    re.IGNORECASE,
)


def claims_a_booking(reply: str) -> bool:
    """Whether this reply tells the patient they have an appointment."""
    return bool(_CLAIMS_A_BOOKING.search(reply))


def no_false_claims(
    reply: str,
    *,
    an_appointment_exists: bool,
    language: str | None,
    fallback_phone: str | None = None,
) -> str:
    """Last resort. Nothing should ever reach this with something to fix.

    Every sentence that tells a patient where their request stands is chosen
    from a recorded outcome (`booking_request.Outcome`) by code that has just
    read the database. That is the mechanism. This is not: matching generated
    text against a list of phrases only ever catches the wordings somebody
    thought to list, in the languages they thought to list them in, and a
    patient who is told in the one unlisted wording that they are booked is
    just as stranded.

    It stays because the model still writes the *other* replies -- the
    answers to questions, where nothing about a booking should appear at all
    -- and a claim surfacing there means something upstream is wrong. So when
    this fires it is treated as a defect: logged at ERROR with the intent of
    being investigated, not as a routine sanitising step.
    """
    if an_appointment_exists or not claims_a_booking(reply):
        return reply
    logger.error(
        "reply_claimed_unbacked_booking",
        extra={"reply_length": len(reply)},
    )
    return booking_request.message_for(
        booking_request.Outcome.DELIVERY_FAILED, language, fallback_phone=fallback_phone
    )


async def _remember_what_was_said(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    message: str,
    state: ConversationState,
) -> patient_profile.Found:
    """Read the message for identity and write it down, before anything else.

    `asked_for_name` is taken from the state: a bare "Asadbek" is a name when
    the assistant just asked for one and something else entirely when it did
    not.
    """
    request_so_far = booking_request.Request(
        name=None, phone=None, reason=state.reason,
        day=state.requested_date, at=state.requested_time,
    )
    asked_for_name = (
        FlowStatus(state.status) is FlowStatus.COLLECTING
        and request_so_far.missing is booking_request.Missing.NAME
    )
    found = patient_profile.read_turn(message, asked_for_name=asked_for_name)
    return await patient_profile.remember(session, user_id=user_id, found=found)


async def advance_booking(
    session: AsyncSession,
    *,
    state: ConversationState,
    message: str,
    profile: patient_profile.Profile,
    intent: Intent,
) -> tuple[ConversationState, str | None]:
    """Move a booking request along, and say so when the clinic cannot.

    Returns the state and, when code can answer the whole turn by itself, the
    reply. The questions this asks are fixed wording (see
    app.services.booking_request) because they are asked in almost every
    conversation and must not drift.
    """
    repo = ConversationStateRepository(session)
    found_when = when_service.read(message)

    # A closed day is refused the moment it is named. Doing this later -- at
    # the point the request is sent -- is how a patient came to agree to a
    # Sunday and be told it had gone through.
    if found_when.day is not None:
        problem = booking_request.check_day(found_when.day)
        if problem is not None:
            return state, (
                f"{problem.reason.capitalize()}. "
                f"{booking_request.next_question(booking_request.Missing.DAY, profile.language)}"
            )

    reason = state.reason
    if reason is None and intent in {Intent.BOOKING_REQUEST, Intent.MEDICAL_QUESTION}:
        # The patient's own words, stored to pass to the front desk. Never
        # read as a diagnosis, and never shown back as one.
        reason = message.strip()[:500]

    state = await repo.save(
        state,
        requested_date=found_when.day if found_when.day is not None else state.requested_date,
        requested_time=found_when.at if found_when.at is not None else state.requested_time,
        reason=reason,
    )

    request = booking_request.assemble(profile, state)
    status = booking_request.next_status(request)
    state = await repo.save(state, status=status)

    if request.is_complete:
        return state, booking_request.confirmation_question(request, profile.language)
    return state, booking_request.next_question(request.missing, profile.language)


async def send_to_the_front_desk(
    session: AsyncSession,
    *,
    state: ConversationState,
    profile: patient_profile.Profile,
    user_id: uuid.UUID,
    source: str,
    fallback_phone: str | None,
) -> tuple[ConversationState, str]:
    """Hand a completed request over, and say only what the database now shows.

    The sentence comes from the delivery's outcome, so the two cannot
    disagree. When the write succeeds there is a row on the dashboard's
    "Lidlar" screen and the patient is told an administrator will ring; when
    it fails there is no row, and the patient is told that and given the
    number. Neither sentence is generated, and neither can be reached without
    the corresponding database result.

    No appointment is created. The clinic books by telephone (see
    app.services.booking_request), so no outcome here says one exists.
    """
    repo = ConversationStateRepository(session)
    request = booking_request.assemble(profile, state)

    delivery = await booking_request.deliver(
        session,
        request=request,
        user_id=user_id,
        conversation_id=state.conversation_id,
        source=source,
    )

    if delivery.outcome is booking_request.Outcome.REQUEST_DELIVERED:
        state = await repo.save(state, status=FlowStatus.REQUEST_SENT)
    else:
        # The flow stays where it was. Nothing was recorded, so the patient
        # has not made a request and the state must not say they have --
        # otherwise a retry would be treated as a duplicate and dropped.
        state = await repo.save(state, status=FlowStatus.AWAITING_CONFIRMATION)

    return state, booking_request.message_for(
        delivery.outcome, profile.language, fallback_phone=fallback_phone
    )


async def prepare(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    message: str,
    source: str = "bot",
    fallback_phone: str | None = None,
) -> tuple[ConversationState, patient_profile.Profile, Intent, str | None]:
    """Everything that happens before a reply is written.

    Returns the state, the profile as it now stands, the intent, and a reply
    when code could answer the whole turn without asking the model.
    """
    await lock_conversation(session, conversation_id)

    repo = ConversationStateRepository(session)
    state = await repo.get_or_create(conversation_id)

    found = await _remember_what_was_said(
        session, user_id=user_id, message=message, state=state
    )
    user = await UserRepository(session).get(user_id)
    profile = patient_profile.load(user) if user is not None else patient_profile.Profile()

    intent = classify(message, state=state)

    if found.conflicting_phone is not None:
        # Two different numbers. Neither is overwritten; the patient is asked
        # which to ring, because only they can say.
        return state, profile, intent, (
            f"Sizda ikkita raqam bor: {profile.phone} va {found.conflicting_phone}. "
            "Qaysi biriga qo'ng'iroq qilaylik?"
        )

    if intent is Intent.BOOKING_CONFIRM:
        state, reply = await send_to_the_front_desk(
            session,
            state=state,
            profile=profile,
            user_id=user_id,
            source=source,
            fallback_phone=fallback_phone,
        )
        return state, profile, intent, reply

    # A message the rules cannot place, arriving while the flow is waiting
    # for something, is the answer to what was asked. "Testbek" is not a
    # booking phrase and never will be; it is a name, and the reason it is
    # being typed is that the assistant asked for one a moment ago.
    #
    # Without this the flow stalled exactly where it was most visible: the
    # name was stored, the patient was answered with nothing, and the next
    # thing they heard was silence. An unrelated question mid-flow is placed
    # by the rules (a price question classifies as one) and still goes to the
    # model, so this only catches the genuinely unplaceable.
    answering_an_open_question = intent is Intent.UNKNOWN and FlowStatus(state.status) in {
        FlowStatus.COLLECTING,
        FlowStatus.AWAITING_DATE,
        FlowStatus.AWAITING_TIME,
    }

    if intent in BOOKING_INTENTS or answering_an_open_question:
        state, reply = await advance_booking(
            session, state=state, message=message, profile=profile, intent=intent
        )
        return state, profile, intent, reply

    return state, profile, intent, None
