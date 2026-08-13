from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class KBSchemaBase(BaseModel):
    question: str
    answer: str
    category: Optional[str] = "general"
    is_active: bool = True


class KBCreate(KBSchemaBase):
    tenant_id: int = 1


class KBUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None


class KBResponse(KBSchemaBase):
    id: int
    tenant_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
