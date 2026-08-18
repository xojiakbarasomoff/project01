"""
Shared phone number extraction and formatting utilities.

Centralised here so that leads.py, telegram.py and tasks.py all use
the same logic.  Any fix to phone parsing only needs to happen once.
"""

import re
from typing import Optional


# ── Pre-compiled patterns ──────────────────────────────────────────────────────
_PHONE_INTL = re.compile(r"\+998\d{9}")
_PHONE_NO_PLUS = re.compile(r"998\d{9}")
# Common Uzbek mobile prefixes (9x, 33, 88, 77, 55, 20) followed by 7 digits
_PHONE_LOCAL = re.compile(r"(9\d|33|88|77|55|20)\d{7}")


def format_phone(raw_phone: Optional[str]) -> Optional[str]:
    """
    Normalise a raw phone string to +998XXXXXXXXX format.

    Returns ``None`` when the input is empty, a placeholder, or unparseable.
    """
    if not raw_phone or raw_phone in ("-", "Telefon kiritilmagan"):
        return None

    cleaned = re.sub(r"[^\d+]", "", str(raw_phone))
    if not cleaned:
        return None

    if cleaned.startswith("+998") and len(cleaned) == 13:
        return cleaned
    if cleaned.startswith("998") and len(cleaned) == 12:
        return "+" + cleaned
    if len(cleaned) == 9 and not cleaned.startswith("+"):
        return "+998" + cleaned
    # Fallback: ensure leading '+'
    if not cleaned.startswith("+"):
        return "+" + cleaned
    return cleaned


def extract_phone_from_text(text: str) -> Optional[str]:
    """
    Scan free-form *original* text for the first phone-number-like substring.

    The search is performed on the raw text so that ``\\b`` word-boundary
    anchors work correctly (unlike searching a digits-only string).
    Returns a normalised +998… string or ``None``.
    """
    if not text:
        return None

    # 1. Try full international format first (+998…)
    m = _PHONE_INTL.search(text)
    if m:
        return m.group(0)

    # 2. Try without leading '+' (998…)
    # Strip non-digit/plus for this check
    digits_only = re.sub(r"[^\d+]", "", text)
    m = _PHONE_NO_PLUS.search(digits_only)
    if m:
        return "+" + m.group(0)

    # 3. Try 9-digit local number on the *original* text (word boundaries work)
    m = _PHONE_LOCAL.search(text)
    if m:
        return "+998" + m.group(0)

    return None
