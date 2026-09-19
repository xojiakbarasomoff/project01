"""What the clinic knows about a patient, kept on their row rather than in a window.

This is the fix for the complaint the clinic made first and loudest: the
assistant asked for a name, was given one, and asked again four messages
later. Nothing was broken in the asking. The name simply was not written
anywhere -- `users.name` and `users.phone` existed and, on Instagram, stayed
NULL forever, because the only thing that ever wrote them was the WhatsApp
path, where the platform hands the number over with every message. Everything
the assistant knew came from the last ten messages, so everything older than
ten messages was forgotten.

So: read each message for a name, a telephone number and a language, and
write what is found onto the patient. Three rules decide how.

**Additive.** A value already on the row is not overwritten by a guess. The
patient typed it or an operator did, and a regular expression is not better
evidence than either.

**A second number is a conflict, not a correction.** Somebody who gives one
number and later another may be giving their husband's, or fixing a typo.
Overwriting silently picks one and loses the other; this reports both and
lets the reply ask which to ring.

**Nothing here is medical.** A reason for the visit is the patient's own
words, stored to pass to the front desk. It is never read as a diagnosis.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.user import UserRepository

logger = logging.getLogger(__name__)

UZBEK_LATIN = "uz-latn"
UZBEK_CYRILLIC = "uz-cyrl"
RUSSIAN = "ru"

# Uzbek mobile numbers: +998 then a two-digit operator code and seven digits.
# The separators people type -- spaces, dashes, brackets, a leading 0 -- are
# stripped before this is applied, so it judges digits only.
_OPERATOR_CODES = frozenset(
    {"20", "33", "50", "55", "61", "62", "65", "66", "67", "69", "70",
     "71", "72", "73", "74", "75", "76", "77", "78", "79", "88", "90",
     "91", "93", "94", "95", "97", "98", "99"}
)

_DIGITS = re.compile(r"[\d+][\d\s\-()+.]{5,}\d")


def normalise_phone(text: str) -> str | None:
    """The first Uzbek mobile number in `text`, as +998XXXXXXXXX, or None.

    Rejected rather than guessed at when the operator code is not one that
    exists: "1234567890" is not a telephone number, and a row carrying it
    means somebody at the front desk dials it and reaches nobody.
    """
    for candidate in _DIGITS.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if digits.startswith("998"):
            digits = digits[3:]
        elif len(digits) == 10 and digits.startswith("0"):
            digits = digits[1:]
        if len(digits) != 9 or digits[:2] not in _OPERATOR_CODES:
            continue
        return "+998" + digits
    return None


# Words that arrive where a name would and are not one. Without this the
# assistant files "Assalomu alaykum" as a patient called Assalomu.
_NOT_A_NAME = frozenset(
    {
        "assalomu", "assalom", "salom", "alaykum", "aleykum", "vaalaykum",
        "ha", "xa", "yoq", "yo'q", "mayli", "rahmat", "raxmat", "zor",
        "qabul", "qabulga", "navbat", "yozilmoqchiman", "kerak", "bor",
        "men", "meni", "mening", "ismim", "otim",
        "здравствуйте", "привет", "да", "нет", "спасибо", "меня", "зовут",
        "ассалом", "салом", "рахмат", "ха", "йок",
    }
)

# "Ismim Asadbek", "Meni Asadbek deb chaqirishadi", "Меня зовут Асадбек".
_NAMED = re.compile(
    r"(?:ism(?:im|i)?|ot(?:im|i)|исм(?:им)?|меня\s+зовут|зовут)"
    r"\s*[-:]?\s*([A-Za-zЀ-ӿ']{3,30})",
    re.IGNORECASE,
)


def looks_like_a_name(text: str) -> bool:
    """Whether a whole message is plausibly just somebody's name.

    Deliberately narrow. A message that is one or two words, all letters, and
    not on the list above. Anything longer is a sentence, and a sentence that
    happens to contain a name is handled by `_NAMED` instead -- guessing at
    names inside free text is how "Qabulga yozilmoqchiman" became a patient.
    """
    words = text.strip().split()
    if not 1 <= len(words) <= 2:
        return False
    for word in words:
        stripped = word.strip(".,!?").lower()
        if not stripped or stripped in _NOT_A_NAME:
            return False
        if not re.fullmatch(r"[A-Za-zЀ-ӿ']{2,30}", stripped):
            return False
    return True


def read_name(text: str, *, asked_for_name: bool) -> str | None:
    """A name in this message, or None.

    `asked_for_name` is what makes a bare "Asadbek" a name: the same word
    arriving unprompted is far more likely to be something else, and the
    assistant only has to be wrong once to start calling somebody "Zor".
    """
    named = _NAMED.search(text)
    if named:
        candidate = named.group(1).strip(".,!?")
        if candidate.lower() not in _NOT_A_NAME:
            return candidate.title()
    if asked_for_name and looks_like_a_name(text):
        return text.strip().strip(".,!?").title()
    return None


_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Letters that exist in Russian but not in Uzbek Cyrillic, and the other way
# round. Present in enough messages to tell the two Cyrillics apart.
_RUSSIAN_ONLY = re.compile(r"[ыэщъ]")
_UZBEK_CYRILLIC_ONLY = re.compile(r"[қғўҳ]")
_RUSSIAN_WORDS = re.compile(
    r"\b(?:здравствуйте|привет|спасибо|хочу|можно"
    r"|записаться|приём|врач|сколько|когда)\b",
    re.IGNORECASE,
)


def read_language(text: str) -> str | None:
    """Which language this message is written in, or None when it cannot tell.

    None matters as much as the answer. "Ok" and "+998901234567" are not
    evidence of anything, and treating them as evidence is what let a
    conversation conducted in Russian flip back to Uzbek on a one-word turn.
    """
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    if cyrillic + latin < 4:
        return None
    if cyrillic > latin:
        if _UZBEK_CYRILLIC_ONLY.search(text):
            return UZBEK_CYRILLIC
        if _RUSSIAN_ONLY.search(text) or _RUSSIAN_WORDS.search(text):
            return RUSSIAN
        return UZBEK_CYRILLIC
    return UZBEK_LATIN


@dataclass(frozen=True)
class Found:
    """What one message turned out to contain."""

    name: str | None = None
    phone: str | None = None
    language: str | None = None
    # A second, different number from one already on the row. Not written
    # over the first; handed back so the reply can ask which one to ring.
    conflicting_phone: str | None = None


def read_turn(text: str, *, asked_for_name: bool = False) -> Found:
    """Everything this one message says about who is writing it."""
    return Found(
        name=read_name(text, asked_for_name=asked_for_name),
        phone=normalise_phone(text),
        language=read_language(text),
    )


@dataclass(frozen=True)
class Profile:
    """What the clinic knows, read back from the patient's row."""

    name: str | None = None
    phone: str | None = None
    language: str | None = None

    @property
    def knows_who_they_are(self) -> bool:
        return bool(self.name and self.phone)


def load(user: User) -> Profile:
    return Profile(name=user.name, phone=user.phone, language=user.preferred_language)


async def remember(
    session: AsyncSession, *, user_id: uuid.UUID, found: Found
) -> Found:
    """Write what `found` adds to the patient's row, and report any conflict.

    Returns `found` with `conflicting_phone` filled in when the message
    carried a number different from the one already stored. Nothing is
    overwritten: the returned conflict is for the reply to raise with the
    patient, which is the only place it can actually be resolved.
    """
    user = await UserRepository(session).get(user_id)
    if user is None:
        return found

    conflicting: str | None = None
    changed = False

    if found.name and not user.name:
        user.name = found.name[:255]
        changed = True
    if found.phone:
        if not user.phone:
            user.phone = found.phone
            changed = True
        elif user.phone != found.phone:
            conflicting = found.phone
    if found.language and user.preferred_language != found.language:
        # Language is the one field a later message may correct: a patient
        # who switches to Russian has switched, and continuing in Uzbek
        # because Uzbek was written down first is the bug, not the fix.
        user.preferred_language = found.language
        changed = True

    if changed:
        await session.flush()
        # No name, number or message text: this line exists to show that
        # memory is being written, not to record what was written.
        logger.info(
            "patient_profile_updated",
            extra={
                "user_id": str(user_id),
                "learned_name": bool(found.name and user.name == found.name),
                "learned_phone": bool(found.phone and user.phone == found.phone),
                "language": user.preferred_language,
            },
        )
    return Found(
        name=found.name,
        phone=found.phone,
        language=found.language,
        conflicting_phone=conflicting,
    )
