"""Turning "ertaga soat 11 da" into a date and a time, in Asia/Tashkent.

The model used to do this, and it got it wrong in the way that matters: a
patient wrote "bugun 16:20" and was told "ertaga 16:20". A language model
reading a relative date has no clock -- it has whatever the prompt said the
date was, and a sentence telling it to add a day. This has a clock, and it
is the only thing in the application allowed to decide what day a patient
meant.

Deliberately conservative. Anything it cannot place it declines to place, and
the caller then asks the patient rather than guessing. A wrong day that is
confidently acted on is far worse than one more question: the patient arrives
on Tuesday for a Wednesday appointment, and nobody finds out until they are
standing at the desk.

`PSA 6.2` is not the sixth of February. Bare numbers are only read as dates
when the message is short enough to be an answer to "qaysi kun?", and never
when the surrounding words say they are something else.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.services.appointment import CLINIC_TIMEZONE

# Monday is 0, matching datetime.weekday().
_WEEKDAYS: dict[str, int] = {
    "dushanba": 0, "душанба": 0, "понедельник": 0,
    "seshanba": 1, "сешанба": 1, "вторник": 1,
    "chorshanba": 2, "чоршанба": 2, "среда": 2, "среду": 2,
    "payshanba": 3, "пайшанба": 3, "четверг": 3,
    "juma": 4, "жума": 4, "пятница": 4, "пятницу": 4,
    "shanba": 5, "шанба": 5, "суббота": 5, "субботу": 5,
    "yakshanba": 6, "якшанба": 6, "воскресенье": 6,
}

_MONTHS: dict[str, int] = {
    "yanvar": 1, "январ": 1, "января": 1,
    "fevral": 2, "феврал": 2, "февраля": 2,
    "mart": 3, "март": 3, "марта": 3,
    "aprel": 4, "апрел": 4, "апреля": 4,
    "may": 5, "мая": 5,
    "iyun": 6, "июн": 6, "июня": 6,
    "iyul": 7, "июл": 7, "июля": 7,
    "avgust": 8, "август": 8, "августа": 8,
    "sentabr": 9, "сентябр": 9, "сентября": 9,
    "oktabr": 10, "октябр": 10, "октября": 10,
    "noyabr": 11, "ноябр": 11, "ноября": 11,
    "dekabr": 12, "декабр": 12, "декабря": 12,
}

_TODAY = re.compile(r"\bbugun\b|\bбугун\b|\bсегодня\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\bertaga\b|\bэртага\b|\bзавтра\b", re.IGNORECASE)
_DAY_AFTER = re.compile(r"\bindinga\b|\bиндинга\b|\bпослезавтра\b", re.IGNORECASE)
# "kelasi shanba", "следующий вторник" -- the one in the week after this.
_NEXT_WEEK = re.compile(r"\bkelasi\b|\bkeyingi\b|\bкеласи\b|\bследующ\w*\b", re.IGNORECASE)

# 16:20, 16.20, 16-20. Anchored so a decimal in "PSA 6.2" cannot match: this
# wants two digits after the separator.
_CLOCK = re.compile(r"\b([01]?\d|2[0-3])[:.\-]([0-5]\d)\b")
# "soat 11 da", "11 larda", "в 11". A bare hour, which needs a word beside it
# saying it is a time -- otherwise every "2 kun" becomes two o'clock.
_BARE_HOUR = re.compile(
    r"(?:soat|соат|в)\s*(\d{1,2})(?:\s*[-]?\s*(?:da|da|larda|да|лар))?\b"
    r"|\b(\d{1,2})\s*(?:da|da\b|larda|да|часов|час)\b",
    re.IGNORECASE,
)

# "25/9", "25.09", "25.09.2026" -- and deliberately not "6.2".
#
# The two separators are treated differently because only one of them is
# ambiguous. Nobody writes a decimal with a slash, so "25/9" is safe with one
# digit either side. A full stop is another matter: "PSA 6.2 chiqdi" is a
# frightened patient reading a lab result, and reading it as the sixth of
# February put a booking on a day nobody had mentioned. So the dotted form
# requires a two-digit month, which every written date has and no decimal
# reading does.
_NUMERIC_DATE = re.compile(
    r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b"
    r"|\b(\d{1,2})\.(\d{2})(?:\.(\d{2,4}))?\b"
)
# "25 sentabr", "25-sentabr".
_DAY_MONTH = re.compile(r"\b(\d{1,2})\s*[-\s]?\s*([a-zA-ZЀ-ӿ]{3,12})\b")


@dataclass(frozen=True)
class When:
    """A day, a time, or both -- whatever the message actually contained."""

    day: date | None = None
    at: time | None = None

    def __bool__(self) -> bool:
        return self.day is not None or self.at is not None


def now() -> datetime:
    """The clinic's own wall clock. Every relative day is measured from here."""
    return datetime.now(CLINIC_TIMEZONE)


def _next_weekday(today: date, weekday: int, *, skip_a_week: bool) -> date:
    ahead = (weekday - today.weekday()) % 7
    # "shanba" said on a Saturday means the one coming, not today: a patient
    # naming a day is naming a day they are not already in the middle of.
    if ahead == 0:
        ahead = 7
    if skip_a_week:
        ahead += 7
    return today + timedelta(days=ahead)


def read_time(text: str) -> time | None:
    """The clock time in `text`, or None."""
    clock = _CLOCK.search(text)
    if clock:
        return time(int(clock.group(1)), int(clock.group(2)))
    bare = _BARE_HOUR.search(text)
    if bare:
        hour = int(bare.group(1) or bare.group(2))
        if 0 <= hour <= 23:
            # A clinic that opens at nine and closes in the evening: "1 da"
            # is one in the afternoon, because there is no one in the morning
            # to mean. Only for hours that are unambiguous in this way.
            if 1 <= hour <= 8:
                hour += 12
            return time(hour, 0)
    return None


def read_day(text: str, *, today: date | None = None) -> date | None:
    """The calendar day `text` refers to, or None when it names none."""
    reference = today or now().date()
    skip_a_week = bool(_NEXT_WEEK.search(text))

    if _TODAY.search(text):
        return reference
    if _TOMORROW.search(text):
        return reference + timedelta(days=1)
    if _DAY_AFTER.search(text):
        return reference + timedelta(days=2)

    lowered = text.lower()
    for word, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{word}", lowered):
            return _next_weekday(reference, weekday, skip_a_week=skip_a_week)

    day_month = _DAY_MONTH.search(text)
    if day_month:
        day_number = int(day_month.group(1))
        word = day_month.group(2).lower()
        for name, month in _MONTHS.items():
            if word.startswith(name[:4]):
                return _resolve(reference, day_number, month)

    numeric = _NUMERIC_DATE.search(text)
    if numeric:
        # Whichever of the two alternatives matched: groups 1-3 are the
        # slashed form, 4-6 the dotted one.
        slashed = numeric.group(1) is not None
        day_number = int(numeric.group(1) if slashed else numeric.group(4))
        month = int(numeric.group(2) if slashed else numeric.group(5))
        year = numeric.group(3) if slashed else numeric.group(6)
        if not 1 <= month <= 12 or not 1 <= day_number <= 31:
            return None
        if year is not None:
            resolved_year = int(year) + (2000 if int(year) < 100 else 0)
            try:
                return date(resolved_year, month, day_number)
            except ValueError:
                return None
        return _resolve(reference, day_number, month)

    return None


def _resolve(reference: date, day_number: int, month: int) -> date | None:
    """A day and month with no year: this year, or next if it has passed.

    Somebody writing "5 yanvar" in December means the January five weeks
    away, not the one eleven months behind them.
    """
    for year in (reference.year, reference.year + 1):
        try:
            candidate = date(year, month, day_number)
        except ValueError:
            return None
        if candidate >= reference:
            return candidate
    return None


def read(text: str, *, today: date | None = None) -> When:
    """Everything this message says about when the patient wants to come."""
    return When(day=read_day(text, today=today), at=read_time(text))


_SPOKEN_DAYS = ("dushanba", "seshanba", "chorshanba", "payshanba", "juma", "shanba", "yakshanba")


def spoken(day: date, *, today: date | None = None) -> str:
    """How the clinic says this date out loud: "ertaga (20.09, shanba)".

    The weekday is in there on purpose. A patient who is told the date and
    the day of the week catches the clinic's mistake before they travel,
    which no amount of internal checking can guarantee on its own.
    """
    reference = today or now().date()
    delta = (day - reference).days
    prefix = {0: "bugun", 1: "ertaga", 2: "indinga"}.get(delta)
    stamp = f"{day.strftime('%d.%m.%Y')}, {_SPOKEN_DAYS[day.weekday()]}"
    return f"{prefix} ({stamp})" if prefix else stamp
