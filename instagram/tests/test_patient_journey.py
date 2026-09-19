"""One patient, start to finish, against a real database.

Everything else in this suite checks a part. This walks the whole thing: a
stranger writes in, says what is wrong, gives a name and a number, names a
day and a time, confirms, and ends up as a row the front desk can work from.
Then they write again, and the clinic does not ask who they are.

Nothing here touches Instagram. `turn.prepare` is called directly, which is
the same function the worker calls, so the flow being exercised is the real
one -- but no Send API call is made and no message reaches anybody. There is
no test patient on a real account, because there is no way to make one that
is not somebody's actual inbox.

The patient details below are invented and stay in this file. Assertions
compare against local variables rather than printing anything, so a failing
run shows the test's own fixtures and not a real number.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_state import FlowStatus
from app.models.lead import LeadStatus
from app.repositories.lead import LeadRepository
from app.repositories.user import UserRepository
from app.services import booking_request, turn
from app.services import when as when_service
from app.services.intent import Intent

# Invented. Not a real Uzbek subscriber: 99 is a valid operator code, and the
# rest is a fixed pattern that exists nowhere.
PATIENT_NAME = "Testbek"
PATIENT_PHONE_TYPED = "99 000 00 11"
PATIENT_PHONE_STORED = "+998990000011"


def _a_weekday_ahead() -> str:
    """A day the clinic could plausibly see somebody, written as a patient would.

    Picked rather than hard-coded so the journey does not depend on which day
    the suite happens to run: "ertaga" is a Sunday once a week.
    """
    day = when_service.now().date()
    for _ in range(8):
        day = day.fromordinal(day.toordinal() + 1)
        if day.weekday() < 5:
            return day.strftime("%d/%m")
    raise AssertionError("no weekday in the next eight days")


async def test_a_new_patient_becomes_a_lead_and_is_remembered(
    db_session: AsyncSession, seed, as_tenant
) -> None:
    conversation_id = seed.a.conversation.id
    user_id = seed.a.user.id

    with as_tenant(seed.tenant_a.id):
        # The clinic knows nothing about this person yet.
        user = await UserRepository(db_session).get(user_id)
        user.name = None
        user.phone = None
        user.preferred_language = None
        await db_session.flush()

        # 1. They say what is wrong and that they want to be seen.
        state, profile, intent, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="Assalomu alaykum, qabulga yozilmoqchiman, belim og'riyapti",
        )
        # A booking request, not a medical question -- the complaint the
        # clinic made loudest.
        assert intent is Intent.BOOKING_REQUEST
        assert profile.language == "uz-latn"
        assert state.reason is not None
        # It listens before it asks who they are.
        assert reply == booking_request.next_question(
            booking_request.Missing.NAME, "uz-latn"
        )

        # 2. The name.
        state, profile, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message=PATIENT_NAME,
        )
        assert profile.name == PATIENT_NAME
        assert reply == booking_request.next_question(
            booking_request.Missing.PHONE, "uz-latn"
        )

        # 3. The number, typed the way people type it.
        state, profile, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message=PATIENT_PHONE_TYPED,
        )
        assert profile.phone == PATIENT_PHONE_STORED

        # 4. A day and a time.
        state, profile, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message=f"{_a_weekday_ahead()} kuni soat 11:00 da",
        )
        assert state.requested_date is not None
        assert state.requested_time is not None
        # Everything is in hand, so it reads the request back instead of
        # sending it.
        assert FlowStatus(state.status) is FlowStatus.AWAITING_CONFIRMATION
        assert "Yuboramizmi" in reply

        # 5. "Ha" -- which means confirm only because something is waiting.
        state, profile, intent, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="ha",
        )
        assert intent is Intent.BOOKING_CONFIRM
        assert FlowStatus(state.status) is FlowStatus.REQUEST_SENT

        # The reply is the delivered-outcome sentence, and it does not claim
        # an appointment exists.
        assert reply == booking_request.message_for(
            booking_request.Outcome.REQUEST_DELIVERED, "uz-latn"
        )
        assert not turn.claims_a_booking(reply)

        # 6. And there is a row the front desk can work from.
        lead = await LeadRepository(db_session).get_open_for_conversation(conversation_id)
        assert lead is not None
        assert lead.patient_name == PATIENT_NAME
        assert lead.phone == PATIENT_PHONE_STORED
        assert lead.status == LeadStatus.NEW.value
        assert lead.convenient_time

        # 7. They write again. Nothing is asked twice.
        _, profile, _, _ = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="Yana bir savolim bor edi",
        )
        assert profile.name == PATIENT_NAME
        assert profile.phone == PATIENT_PHONE_STORED


async def test_confirming_twice_does_not_queue_the_patient_twice(
    db_session: AsyncSession, seed, as_tenant
) -> None:
    """A patient who taps "ha" twice, or whose job is retried, must not end
    up being rung twice by two different people.
    """
    conversation_id = seed.a.conversation.id
    user_id = seed.a.user.id

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get(user_id)
        user.name = PATIENT_NAME
        user.phone = PATIENT_PHONE_STORED
        user.preferred_language = "uz-latn"
        await db_session.flush()

        for message in (
            "qabulga yozilmoqchiman",
            f"{_a_weekday_ahead()} kuni soat 11:00 da",
            "ha",
            "ha",
        ):
            await turn.prepare(
                db_session,
                conversation_id=conversation_id,
                user_id=user_id,
                message=message,
            )

        leads = await LeadRepository(db_session).list_recent()
        mine = [lead for lead in leads if lead.conversation_id == conversation_id]
        assert len(mine) == 1


async def test_a_failed_handover_never_tells_the_patient_to_wait(
    db_session: AsyncSession, seed, as_tenant, monkeypatch
) -> None:
    """The row could not be written, so nobody will ring. The reply has to
    say that, and the flow must stay where it was so a retry still works.
    """
    conversation_id = seed.a.conversation.id
    user_id = seed.a.user.id

    async def explode(*args, **kwargs):
        raise RuntimeError("database unavailable")

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get(user_id)
        user.name = PATIENT_NAME
        user.phone = PATIENT_PHONE_STORED
        user.preferred_language = "uz-latn"
        await db_session.flush()

        await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="qabulga yozilmoqchiman",
        )
        await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message=f"{_a_weekday_ahead()} kuni soat 11:00 da",
        )

        monkeypatch.setattr(LeadRepository, "create", explode)
        monkeypatch.setattr(LeadRepository, "get_open_for_conversation", explode)

        state, _, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="ha",
        )

    assert reply == booking_request.message_for(
        booking_request.Outcome.DELIVERY_FAILED, "uz-latn", fallback_phone=None
    )
    assert "kuting" not in reply
    # Still awaiting confirmation, so the patient saying "ha" again is a
    # fresh attempt rather than a duplicate to be dropped.
    assert FlowStatus(state.status) is FlowStatus.AWAITING_CONFIRMATION


async def test_a_russian_speaker_stays_in_russian(
    db_session: AsyncSession, seed, as_tenant
) -> None:
    conversation_id = seed.a.conversation.id
    user_id = seed.a.user.id

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get(user_id)
        user.name = user.phone = user.preferred_language = None
        await db_session.flush()

        _, profile, _, _ = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="Здравствуйте, хочу записаться на приём",
        )
        assert profile.language == "ru"

        # A one-word turn carries no evidence of language and must not
        # change it back.
        _, profile, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="Ok",
        )
        assert profile.language == "ru"
        # "Ok" answers the open question, so the flow -- not the model --
        # produces the next one, and it is in Russian.
        assert reply in {
            booking_request.next_question(missing, "ru")
            for missing in booking_request.Missing
        }


async def test_a_sheets_outage_cannot_undo_the_lead(
    db_session: AsyncSession, seed, as_tenant, monkeypatch
) -> None:
    """The spreadsheet is a copy. The `leads` row is the record.

    An owner's Google mirror failing must not roll back, delete or block the
    row the front desk works from -- nor change what the patient was told.
    The mirror runs after the send, outside the write, and its errors are
    logged rather than raised.
    """
    from app.services import sheets

    async def explode(*args, **kwargs):
        raise RuntimeError("Google is down")

    monkeypatch.setattr(sheets, "mirror_lead", explode)
    monkeypatch.setattr(sheets, "mirror_appointment", explode)

    conversation_id = seed.a.conversation.id
    user_id = seed.a.user.id

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get(user_id)
        user.name, user.phone, user.preferred_language = (
            PATIENT_NAME,
            PATIENT_PHONE_STORED,
            "uz-latn",
        )
        await db_session.flush()

        for message in ("qabulga yozilmoqchiman", f"{_a_weekday_ahead()} kuni soat 11:00 da"):
            await turn.prepare(
                db_session,
                conversation_id=conversation_id,
                user_id=user_id,
                message=message,
            )
        state, _, _, reply = await turn.prepare(
            db_session,
            conversation_id=conversation_id,
            user_id=user_id,
            message="ha",
        )

        lead = await LeadRepository(db_session).get_open_for_conversation(conversation_id)

    # The row survives the outage, and so does the sentence the patient got.
    assert lead is not None
    assert lead.phone == PATIENT_PHONE_STORED
    assert FlowStatus(state.status) is FlowStatus.REQUEST_SENT
    assert reply == booking_request.message_for(
        booking_request.Outcome.REQUEST_DELIVERED, "uz-latn"
    )
