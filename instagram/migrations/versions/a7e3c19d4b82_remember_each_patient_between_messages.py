"""Remember each patient between messages

Three additions, no removals and no rewrites of existing rows.

  * users.preferred_language -- the language a patient chose, so it survives
    past the end of the context window.
  * conversations.summary / summarised_message_count -- a rolling description
    of a long conversation, which is a convenience and never a source of
    facts.
  * conversation_states -- where a booking request has got to, one row per
    conversation.

Every column is nullable or has a server default, so the deploy is safe in
either order: the running code ignores what it does not know about, and the
new code finds defaults on rows written before it.

Revision ID: a7e3c19d4b82
Revises: d5b7e04a1c63
Create Date: 2026-09-19

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a7e3c19d4b82"
down_revision: str | None = "d5b7e04a1c63"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("preferred_language", sa.String(length=16), nullable=True))
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "conversations",
        sa.Column(
            "summarised_message_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )

    op.create_table(
        "conversation_states",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=40), server_default=sa.text("'idle'"), nullable=False),
        sa.Column("requested_date", sa.Date(), nullable=True),
        sa.Column("requested_time", sa.Time(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("appointment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["appointment_id"], ["appointments.id"]),
        sa.PrimaryKeyConstraint("id"),
        # One flow per conversation. Two webhook jobs racing to open a state
        # both find none and both insert; the constraint is what decides
        # there is one, rather than a lucky interleaving.
        sa.UniqueConstraint("conversation_id", name="uq_conversation_states_conversation"),
    )
    op.create_index(
        "ix_conversation_states_tenant_id", "conversation_states", ["tenant_id"], unique=False
    )
    op.create_index(
        "ix_conversation_states_conversation_id",
        "conversation_states",
        ["conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_states_conversation_id", table_name="conversation_states")
    op.drop_index("ix_conversation_states_tenant_id", table_name="conversation_states")
    op.drop_table("conversation_states")
    op.drop_column("conversations", "summarised_message_count")
    op.drop_column("conversations", "summary")
    op.drop_column("users", "preferred_language")
