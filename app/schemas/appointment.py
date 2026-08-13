from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class AppointmentBase(BaseModel):
    patient_name: str
    patient_phone: str
    doctor_name: Optional[str] = "Stomatolog Shifokor"
    appointment_time: Optional[datetime] = None
    notes: Optional[str] = None


class AppointmentCreate(AppointmentBase):
    tenant_id: int = 1
    user_id: Optional[int] = None


class AppointmentUpdate(BaseModel):
    patient_name: Optional[str] = None
    patient_phone: Optional[str] = None
    doctor_name: Optional[str] = None
    appointment_time: Optional[datetime] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class AppointmentResponse(AppointmentBase):
    id: int
    tenant_id: int
    user_id: Optional[int] = None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
