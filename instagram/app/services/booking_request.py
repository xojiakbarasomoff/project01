"""Collecting a request to be seen, and handing it to the front desk.

The clinic moved its bookings to the telephone in September, deliberately:
patients are seen live and the time is agreed by voice. The appointment book
is still there and still correct -- staff work it from the dashboard -- but
nothing automatic writes to it, and this does not change that. Re-enabling
automatic booking is the clinic's decision, not a side effect of adding
memory.

So this collects. Name, telephone, the patient's own words about why, and
the day and time they would like. When all of it is in hand the request goes
to the front desk and the patient is told exactly that:

    "So'rovingiz yuborildi. Qabul vaqti tasdiqlanishini kuting."

Never "qabulingiz tasdiqlandi". The distinction is the entire point. A
patient who is told they are booked stops thinking about it, and when nobody
rang and nothing was in the diary, they arrive to find they are not expected.
`confirmed_wording_is_allowed` is the one switch that changes this, and it is
False until the clinic turns automatic booking back on.

What is collected lives in two places, and which is which matters. Name,
telephone and language are facts about the patient and belong on `users`,
where they outlive this request. The day, the time and the reason belong to
this request and live on `conversation_states`, where abandoning the request
clears them without costing the clinic the contact details.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import date, time
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_state import ConversationState, FlowStatus
from app.models.lead import LeadStatus
from app.repositories.lead import LeadRepository
from app.services import clinic_schedule
from app.services import when as when_service
from app.services.patient_profile import Profile

logger = logging.getLogger(__name__)

# Until the clinic turns automatic booking back on, nothing may tell a
# patient their appointment is confirmed. This is a module constant rather
# than a setting because flipping it is a change of behaviour that should be
# reviewed and deployed, not typed into a dashboard at midnight.
CONFIRMED_WORDING_IS_ALLOWED = False


class Missing(StrEnum):
    """Which part of a request is not in hand yet, in the order to ask for it."""

    NAME = "name"
    PHONE = "phone"
    REASON = "reason"
    DAY = "day"
    TIME = "time"
    NOTHING = "nothing"


@dataclass(frozen=True)
class Request:
    """A request as it currently stands, assembled from profile and state."""

    name: str | None
    phone: str | None
    reason: str | None
    day: date | None
    at: time | None

    @property
    def missing(self) -> Missing:
        """The next thing to ask for, in the order the doctor asks for it.

        Taken from the clinic's own transcripts rather than invented: the
        doctor opens with "Eshitaman", learns what the trouble is, and only
        then asks who is writing and where they are. Asking for a name before
        listening is the thing patients complained about.
        """
        if not self.reason:
            return Missing.REASON
        if not self.name:
            return Missing.NAME
        if not self.phone:
            return Missing.PHONE
        if self.day is None:
            return Missing.DAY
        if self.at is None:
            return Missing.TIME
        return Missing.NOTHING

    @property
    def is_complete(self) -> bool:
        return self.missing is Missing.NOTHING


def assemble(profile: Profile, state: ConversationState) -> Request:
    """What the clinic has, read from the two places that hold it."""
    return Request(
        name=profile.name,
        phone=profile.phone,
        reason=state.reason,
        day=state.requested_date,
        at=state.requested_time,
    )


@dataclass(frozen=True)
class DayProblem:
    """A day the clinic cannot accept, and why, in the patient's language."""

    reason: str


def check_day(day: date, *, today: date | None = None) -> DayProblem | None:
    """Whether the clinic can see somebody on this day at all.

    Checked the moment the patient names the day, not when the request is
    sent: somebody who says "ertaga" on a Saturday should hear that the
    clinic is shut straight away, not agree to a whole request first.

    A day in the past is refused on arithmetic alone. Everything else needs
    the clinic to have said when it opens, and when it has not, this returns
    None -- no objection raised, because raising one would mean inventing the
    opening days. The request then goes to the front desk, who know. Telling
    a patient the clinic is closed on a day it is open loses them; passing an
    awkward request to a human does not.
    """
    reference = today or when_service.now().date()
    if day < reference:
        return DayProblem("o'tgan kun")

    schedule = clinic_schedule.load_or_none()
    if schedule is None:
        return None
    if not schedule.is_open_on(day):
        return DayProblem("u kuni klinika yopiq")
    if (day - reference).days > schedule.horizon_days:
        return DayProblem(f"{schedule.horizon_days} kundan uzoqqa yozib bo'lmaydi")
    return None


def next_question(missing: Missing, language: str | None) -> str:
    """What to ask next, in the patient's own language.

    Fixed wording rather than generated. These five sentences are asked in
    nearly every conversation, they must never drift, and a model asked to
    produce them will eventually produce two questions at once -- which is
    the thing the clinic's own style never does.
    """
    questions = _QUESTIONS.get(language or "uz-latn", _QUESTIONS["uz-latn"])
    return questions[missing]


_QUESTIONS: dict[str, dict[Missing, str]] = {
    "uz-latn": {
        Missing.REASON: "Qanday muammo bo'yicha murojaat qilyapsiz?",
        Missing.NAME: "Ismingizni ayta olasizmi?",
        Missing.PHONE: "Telefon raqamingizni qoldiring.",
        Missing.DAY: "Qaysi kun sizga qulay?",
        Missing.TIME: "Soat nechada kelsangiz qulay bo'ladi?",
        Missing.NOTHING: "",
    },
    "uz-cyrl": {
        Missing.REASON: "Қандай муаммо бўйича мурожаат қиляпсиз?",
        Missing.NAME: "Исмингизни айта оласизми?",
        Missing.PHONE: "Телефон рақамингизни қолдиринг.",
        Missing.DAY: "Қайси кун сизга қулай?",
        Missing.TIME: "Соат нечада келсангиз қулай бўлади?",
        Missing.NOTHING: "",
    },
    "ru": {
        Missing.REASON: "С каким вопросом вы обращаетесь?",
        Missing.NAME: "Как вас зовут?",
        Missing.PHONE: "Оставьте, пожалуйста, ваш номер телефона.",
        Missing.DAY: "Какой день вам удобен?",
        Missing.TIME: "Во сколько вам удобно подойти?",
        Missing.NOTHING: "",
    },
}


def confirmation_question(request: Request, language: str | None) -> str:
    """Read the whole request back and ask for a yes before sending it.

    Everything the clinic will act on is in this sentence, including the
    weekday, so a patient who is about to be put down for the wrong day sees
    it before anybody travels.
    """
    day = when_service.spoken(request.day) if request.day else ""
    at = request.at.strftime("%H:%M") if request.at else ""
    if language == "ru":
        return f"Записываю заявку: {day}, {at}. Отправляем?"
    if language == "uz-cyrl":
        return f"Сўров: {day}, соат {at}. Юборамизми?"
    return f"So'rov: {day}, soat {at}. Yuboramizmi?"


class Outcome(StrEnum):
    """What actually happened in the database, which is what the patient is told.

    The reply is selected from this, not filtered out of whatever the model
    wrote. Screening generated text for the phrase "yozib qo'ydim" only
    catches the wordings somebody thought of; choosing the sentence from a
    recorded outcome means there is no wording to catch, because the sentence
    was never generated in the first place.

    There is no member meaning "an appointment was created" while automatic
    booking is off. When the clinic turns it on, BOOKING_CREATED is added
    here and given its own sentence -- and it will be set by the code that
    committed the row, or not at all.
    """

    #: Still gathering; nothing has been promised.
    COLLECTING = "collecting"
    #: Read back to the patient, waiting for a yes.
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    #: A row exists in `leads`; the front desk can see it.
    REQUEST_DELIVERED = "request_delivered"
    #: Nothing was written. The patient must not be told to wait for a call.
    DELIVERY_FAILED = "delivery_failed"
    #: The clinic cannot take that day.
    DAY_REFUSED = "day_refused"


def message_for(
    outcome: Outcome, language: str | None, *, fallback_phone: str | None = None
) -> str:
    """The sentence that goes with a recorded outcome.

    Fixed text per outcome per language. Nothing here is generated, so
    nothing here can claim more than happened.
    """
    return _OUTCOME_MESSAGES[outcome][language or "uz-latn"].format(
        phone=fallback_phone or ""
    ).strip()


_OUTCOME_MESSAGES: dict[Outcome, dict[str, str]] = {
    Outcome.REQUEST_DELIVERED: {
        "uz-latn": (
            "Ma'lumotlaringiz qabul qilindi va administratorga uzatildi. "
            "Qabul vaqtini administrator tasdiqlaydi va sizga qo'ng'iroq qiladi."
        ),
        "uz-cyrl": (
            "Маълумотларингиз қабул қилинди ва администраторга узатилди. "
            "Қабул вақтини администратор тасдиқлайди ва сизга қўнғироқ қилади."
        ),
        "ru": (
            "Ваши данные приняты и переданы администратору. "
            "Время приёма подтвердит администратор и перезвонит вам."
        ),
    },
    # Says plainly that nothing was recorded, and gives the patient a way
    # through that does not depend on this system working. Telling somebody
    # to wait for a call that nobody will make is the failure this exists to
    # prevent.
    Outcome.DELIVERY_FAILED: {
        "uz-latn": (
            "Kechirasiz, ma'lumotlaringizni saqlashda nosozlik bo'ldi. "
            "Iltimos, {phone} raqamiga qo'ng'iroq qiling."
        ),
        "uz-cyrl": (
            "Кечирасиз, маълумотларингизни сақлашда носозлик бўлди. "
            "Илтимос, {phone} рақамига қўнғироқ қилинг."
        ),
        "ru": (
            "Извините, произошёл сбой при сохранении ваших данных. "
            "Пожалуйста, позвоните по номеру {phone}."
        ),
    },
}


def next_status(request: Request) -> FlowStatus:
    """Where the flow sits given what is still missing."""
    match request.missing:
        case Missing.DAY:
            return FlowStatus.AWAITING_DATE
        case Missing.TIME:
            return FlowStatus.AWAITING_TIME
        case Missing.NOTHING:
            return FlowStatus.AWAITING_CONFIRMATION
        case _:
            return FlowStatus.COLLECTING


@dataclass(frozen=True)
class Delivery:
    """What happened when the request was handed over."""

    outcome: Outcome
    lead_id: uuid.UUID | None = None
    error: str | None = None


async def deliver(
    session: AsyncSession,
    *,
    request: Request,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    source: str,
) -> Delivery:
    """Write the request where the front desk will see it, and report what happened.

    The destination is `leads`, which already exists, is already on the
    dashboard's "Lidlar" screen and is already mirrored into the clinic's
    spreadsheet. Nothing new had to be built for a human to see this; what was
    missing was anything actually writing the row, so the assistant had been
    telling patients their request was "sent" when the only thing sent was a
    line to a log file.

    Failure is reported, not swallowed. A patient whose row could not be
    written is told so and given the telephone number instead -- see
    Outcome.DELIVERY_FAILED. The alternative, which is what the previous
    version did by construction, is a patient waiting for a call that nobody
    was ever asked to make.

    Idempotent per conversation: a patient who confirms twice updates the one
    lead rather than putting themselves in the callback queue twice.
    """
    repo = LeadRepository(session)
    convenient = ""
    if request.day is not None:
        convenient = when_service.spoken(request.day)
        if request.at is not None:
            convenient += f", {request.at:%H:%M}"

    try:
        existing = await repo.get_open_for_conversation(conversation_id)
        if existing is not None:
            lead = await repo.update(
                existing,
                patient_name=request.name,
                phone=request.phone,
                topic=(request.reason or "")[:255] or None,
                convenient_time=convenient[:255] or None,
            )
        else:
            lead = await repo.create(
                user_id=user_id,
                conversation_id=conversation_id,
                patient_name=request.name,
                phone=request.phone,
                topic=(request.reason or "")[:255] or None,
                convenient_time=convenient[:255] or None,
                status=LeadStatus.NEW.value,
                notes=f"Instagram orqali so'rov ({source})",
            )
        # Flushed, not committed: this belongs in the same transaction as the
        # conversation state that says the request was sent, so the two can
        # never disagree about whether it happened.
        await session.flush()
    except Exception as exc:  # noqa: BLE001 - the outcome is the point, not the type
        # Not re-raised. The caller still has to answer the patient, and the
        # honest answer depends on knowing this failed.
        logger.exception(
            "booking_request_delivery_failed",
            extra={"conversation_id": str(conversation_id)},
        )
        return Delivery(outcome=Outcome.DELIVERY_FAILED, error=type(exc).__name__)

    logger.info(
        "booking_request_delivered",
        extra={
            "conversation_id": str(conversation_id),
            "lead_id": str(lead.id),
            # The day and time only. The name and number are on the row an
            # operator opens, not in a log line that may leave the machine.
            "convenient_time": convenient,
        },
    )
    return Delivery(outcome=Outcome.REQUEST_DELIVERED, lead_id=lead.id)
