"""
Domain-wide constants and enumerations for AIMED.

Using a Python str-Enum prevents typos and makes refactoring safer
than scattering raw strings across views, workers and templates.
"""

from enum import Enum


class LeadStatus(str, Enum):
    YANGI = "yangi"
    ALOQADA = "aloqada"
    BEKOR = "bekor"


# Sentinel value — used when a lead has no phone on record.
# Prefer storing NULL in the database; this constant is only for
# display / export layers.
PHONE_MISSING_DISPLAY = "Telefon kiritilmagan"

# Default topic for auto-synced Telegram leads
DEFAULT_TOPIC = "Telegram bot orqali murojaat"
