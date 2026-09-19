import uuid
from datetime import date, time
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.models.conversation_state import ConversationState, FlowStatus
from app.repositories.base import TenantScopedRepository


class StaleStateError(Exception):
    """Raised when a save is built on a version somebody else has moved past.

    The advisory lock in app.services.turn should make this impossible within
    one process. It is checked anyway: locks are advisory, deployments overlap,
    and the failure this prevents -- one job's idea of the flow silently
    overwriting another's -- is invisible until a patient is told something
    that contradicts what they were told a second earlier.
    """


# Distinguishes "leave this field alone" from "set this field to None", which
# a default of None cannot: clearing a requested time is a real thing the
# caller needs to do when the patient changes their mind.
_UNSET: Any = object()


class ConversationStateRepository(TenantScopedRepository[ConversationState]):
    model = ConversationState

    async def get_for_conversation(self, conversation_id: uuid.UUID) -> ConversationState | None:
        result = await self.session.execute(
            select(ConversationState).where(
                ConversationState.conversation_id == conversation_id,
                ConversationState.tenant_id == self._resolve_tenant_id(None),
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, conversation_id: uuid.UUID) -> ConversationState:
        """This conversation's state row, opened on first need.

        Races the same way `_get_or_create_user` does, and for the same
        reason: two bubbles from one patient can be in flight at once, both
        find no row and both insert. The unique constraint decides.
        """
        existing = await self.get_for_conversation(conversation_id)
        if existing is not None:
            return existing
        try:
            async with self.session.begin_nested():
                created = ConversationState(
                    tenant_id=self._resolve_tenant_id(None),
                    conversation_id=conversation_id,
                    status=FlowStatus.IDLE.value,
                )
                self.session.add(created)
                await self.session.flush()
                return created
        except IntegrityError:
            raced = await self.get_for_conversation(conversation_id)
            if raced is None:
                raise
            return raced

    async def save(
        self,
        state: ConversationState,
        *,
        status: FlowStatus | None = None,
        requested_date: date | None = _UNSET,
        requested_time: time | None = _UNSET,
        reason: str | None = _UNSET,
        appointment_id: uuid.UUID | None = _UNSET,
    ) -> ConversationState:
        """Apply the fields given, bumping the version, or refuse if stale."""
        values: dict[str, Any] = {"version": state.version + 1}
        if status is not None:
            values["status"] = status.value
        if requested_date is not _UNSET:
            values["requested_date"] = requested_date
        if requested_time is not _UNSET:
            values["requested_time"] = requested_time
        if reason is not _UNSET:
            values["reason"] = reason
        if appointment_id is not _UNSET:
            values["appointment_id"] = appointment_id

        result = await self.session.execute(
            update(ConversationState)
            .where(
                ConversationState.id == state.id,
                ConversationState.version == state.version,
            )
            .values(**values)
            .returning(ConversationState.id)
        )
        if result.scalar_one_or_none() is None:
            raise StaleStateError(
                f"conversation_state {state.id} moved past version {state.version}"
            )
        await self.session.refresh(state)
        return state

    async def clear_flow(self, state: ConversationState) -> ConversationState:
        """Put the flow back to idle, forgetting only the request.

        The patient's name, number and language are on `users` and are not
        touched here. An abandoned booking must never cost the clinic the
        contact details it already collected.
        """
        return await self.save(
            state,
            status=FlowStatus.IDLE,
            requested_date=None,
            requested_time=None,
            reason=None,
        )
