import uuid
from datetime import date, datetime, time
from enum import StrEnum

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FlowStatus(StrEnum):
    """Where a conversation has got to, as the clinic's code sees it.

    This is the memory that a ten-message context window cannot hold. A
    patient who gave their name on Monday and comes back on Friday is in
    the same place they left off, because the place is a row rather than
    something inferred from whatever text happens to still be in the window.
    """

    IDLE = "idle"
    # Collecting the four things a request needs: name, phone, reason, time.
    COLLECTING = "collecting"
    AWAITING_DATE = "awaiting_date"
    AWAITING_TIME = "awaiting_time"
    # Everything is in hand and the patient has been asked to confirm.
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    # Sent to the front desk; the clinic decides the time.
    REQUEST_SENT = "request_sent"
    AWAITING_CANCEL_CONFIRM = "awaiting_cancel_confirm"


class ConversationState(Base):
    """One row per conversation: what the clinic knows and what it is waiting for.

    Separate from `conversations` deliberately. That table is the transcript's
    spine -- who, which channel, is the bot on -- and is read on every path in
    the application. This is scratch space for one flow, written on nearly
    every inbound message, and keeping the two apart means a booking in
    progress cannot lock rows that the dashboard's conversation list needs.

    Nothing here is a fact about the patient. Name, telephone and language
    live on `users`, where they outlive the flow that collected them; what is
    here is a request being assembled, and it is cleared when the request is
    done. A patient's details must never disappear because a booking was
    abandoned halfway.
    """

    __tablename__ = "conversation_states"
    __table_args__ = (
        UniqueConstraint("conversation_id", name="uq_conversation_states_conversation"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(40), nullable=False, server_default=text("'idle'")
    )
    # The day and the time the patient asked for, kept apart because they
    # arrive apart: "shanba" on one message, "11 da" on the next. Stored as
    # local clinic date/time -- app.services.when is what turned the words
    # into them, and it is the only thing that does.
    requested_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    requested_time: Mapped[time | None] = mapped_column(Time, nullable=True)
    # Why they want to come, in their own words. Goes to the front desk with
    # the request; never interpreted as a diagnosis.
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The appointment this flow is about, once the clinic has made one.
    appointment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("appointments.id"), nullable=True
    )
    # Optimistic locking. Two webhook jobs for the same conversation are
    # already serialised by an advisory lock (app.services.turn), and this is
    # the second line: a save built on a state somebody else has since moved
    # on from is refused rather than silently overwriting it.
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=text("now()"),
    )
