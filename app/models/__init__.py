from app.db.base import Base
from app.models.domain import (
    Tenant,
    Channel,
    User,
    Conversation,
    Message,
    KnowledgeBase,
    Operator,
    Appointment
)

__all__ = [
    "Base",
    "Tenant",
    "Channel",
    "User",
    "Conversation",
    "Message",
    "KnowledgeBase",
    "Operator",
    "Appointment"
]
