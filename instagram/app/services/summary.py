"""A short description of a long conversation, kept current as it grows.

The context window holds ten turns. A conversation that runs longer than
that loses its beginning, and the beginning is usually where the patient said
what was wrong. This puts a few sentences in front of the model describing
what has happened, so the assistant stops behaving as though every
conversation started four messages ago.

**It is built from columns, not from the transcript.** That is the whole
design. An LLM asked to summarise a chat produces something fluent and
occasionally wrong -- a number off by a digit, a day that was discussed but
never agreed -- and once that text is in the prompt nothing can tell it from
the truth. So the summary is assembled from the fields the clinic already
decided: the patient's row, the flow's row, how many messages there have
been. Every clause in it is a value read from the database a moment earlier.

Which makes the rule the clinic asked for hold by construction rather than
by discipline: nothing is ever *stored only in* the summary. Delete the
summary and nothing is lost, because every fact in it lives somewhere that
the reply logic reads directly. The summary is a convenience for the model's
attention, never a source.

The counter tells the caller when to refresh: there is no point rewriting
this on every message when nothing it describes has changed.
"""

from app.models.conversation import Conversation
from app.models.conversation_state import ConversationState, FlowStatus
from app.services import when as when_service
from app.services.patient_profile import Profile

# Below this many messages the window still holds the whole conversation and
# a summary would only repeat what the model can already read.
SUMMARISE_AFTER_MESSAGES = 8

# How many new messages may arrive before it is rebuilt. Small, because it is
# free to rebuild -- no model call -- and a stale line about a booking that
# has since been sent is worse than no line at all.
REFRESH_EVERY_MESSAGES = 4

_STATUS_WORDS = {
    FlowStatus.IDLE: "hozircha qabul so'rovi ochilmagan",
    FlowStatus.COLLECTING: "qabul so'rovi to'ldirilmoqda",
    FlowStatus.AWAITING_DATE: "qaysi kun kelishi so'ralgan",
    FlowStatus.AWAITING_TIME: "soat nechada kelishi so'ralgan",
    FlowStatus.AWAITING_CONFIRMATION: "so'rov tasdiqlanishi kutilmoqda",
    FlowStatus.REQUEST_SENT: "so'rov administratorga yuborilgan",
    FlowStatus.AWAITING_CANCEL_CONFIRM: "bekor qilish tasdiqlanishi kutilmoqda",
}


def compose(profile: Profile, state: ConversationState, *, message_count: int) -> str:
    """The summary as it should read right now.

    Deliberately terse and deliberately boring. It goes in front of a model
    that is about to write one or two sentences, and a paragraph of narrative
    would crowd out the retrieved FAQ that answers what was actually asked.
    """
    parts: list[str] = [f"Suhbat {message_count} ta xabardan iborat."]

    known: list[str] = []
    if profile.name:
        known.append("ismi ma'lum")
    if profile.phone:
        known.append("telefon raqami ma'lum")
    if known:
        # What is known, not what it is. The values themselves are read from
        # `users` by the code that needs them; repeating them here would put
        # a patient's telephone number into every prompt for no purpose.
        parts.append("Bemorning " + " va ".join(known) + " — qayta so'ramang.")

    if state.reason:
        parts.append(f"Murojaat sababi: {state.reason[:120]}")

    status = FlowStatus(state.status)
    parts.append(_STATUS_WORDS.get(status, str(status)).capitalize() + ".")

    if state.requested_date is not None:
        day = when_service.spoken(state.requested_date)
        at = state.requested_time.strftime("%H:%M") if state.requested_time else None
        parts.append(f"So'ralgan vaqt: {day}" + (f", soat {at}" if at else ""))

    return " ".join(parts)


def needs_refresh(conversation: Conversation, *, message_count: int) -> bool:
    """Whether the stored summary is far enough behind to be worth rebuilding."""
    if message_count < SUMMARISE_AFTER_MESSAGES:
        return False
    if conversation.summary is None:
        return True
    return message_count - conversation.summarised_message_count >= REFRESH_EVERY_MESSAGES


def update(
    conversation: Conversation,
    profile: Profile,
    state: ConversationState,
    *,
    message_count: int,
) -> bool:
    """Rebuild the summary on the conversation if it is due. Returns whether
    anything changed, so the caller only flushes when there is something to
    write.
    """
    if not needs_refresh(conversation, message_count=message_count):
        return False
    conversation.summary = compose(profile, state, message_count=message_count)
    conversation.summarised_message_count = message_count
    return True
