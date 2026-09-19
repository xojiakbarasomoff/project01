import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Conversation(Base):
    __tablename__ = "conversations"
    # At most one *open* conversation per patient, enforced partially so a
    # closed one never blocks the next. Same role as the users constraint
    # above: it settles the concurrent-first-contact race rather than
    # leaving a patient split across two transcripts.
    __table_args__ = (
        Index(
            "uq_conversations_open_per_user",
            "tenant_id",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'open'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    is_bot_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    # A few sentences describing what this conversation has been about, kept
    # current as it grows past what the context window can carry.
    #
    # A summary is a convenience, never a source. Anything the clinic will
    # act on -- the patient's name, their number, the day they asked for --
    # is a column on `users` or `conversation_states`, read from there and
    # not from this text. The rule is the whole reason the summary is safe
    # to keep: it can be wrong, or stale, or quietly lossy, and nothing
    # important depends on it.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many messages of this conversation the summary already covers, so
    # the next update summarises what has happened since rather than the
    # whole transcript again.
    summarised_message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=func.now(), nullable=False
    )
