"""When the clinic is open -- read from configuration, never guessed.

Three numbers decide what a patient is told: which days the clinic opens,
between which hours, and how far ahead it will take a booking. Two of them
were configured and one was not, while all three sat in the code as constants
nobody had checked against the clinic:

  * `WORK_END` was 19:00 from the first import, while every doctor in
    data/doctors.json works until 18:00.
  * There was no weekday rule at all, so Sunday was bookable.
  * Two different horizons existed, 2 days in one module and 14 in another.

The temptation is to pick the values that look right. That is how a patient
gets told the clinic is open on a Saturday it is shut, which is worse than
being told the clinic does not know -- one wastes a message, the other
wastes a journey.

So: read them from settings. When they are missing, say so loudly and let
the caller decide what it can still do safely. `ScheduleNotConfiguredError`
is not an error state to be swallowed; it is the clinic not having told us
something, and the assistant's correct response to it is to stop making
claims about days and times and let the front desk answer.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, time

from app.core.config import Settings, get_settings
from app.services.appointment import DEFAULT_SEARCH_HORIZON_DAYS

logger = logging.getLogger(__name__)


class ScheduleNotConfiguredError(Exception):
    """The clinic has not said when it is open.

    Carries the name of the variable that would fix it, because this is a
    deployment mistake and the message is the whole remedy.
    """


# Two times, however the clinic wrote the join between them.
#
# The first version of this insisted on a dash, and the value actually in
# production is "Dushanbadan shanbagacha 09:00 dan 18:00 gacha" -- the
# clinic's own sentence, written for a patient to read, not for a parser. It
# failed to match, and the assistant fell back to saying nothing about hours
# at all while the clinic had told it twice.
#
# So: find two clock times and accept whatever sits between them, as long as
# it is short enough to be a join ("-", "dan", "to", "до") rather than a
# whole other clause.
_HOURS = re.compile(r"(\d{1,2})[:.](\d{2})\D{0,12}?(\d{1,2})[:.](\d{2})")

# The working week, as the clinic writes it inside that same sentence.
# "Dushanbadan shanbagacha" is Monday-to-Saturday; the spans below are the
# forms that actually appear rather than every form imaginable.
_WEEK_SPANS: list[tuple[str, tuple[int, int]]] = [
    (r"dushanba\w*\s+(?:dan\s+)?shanba\w*", (0, 5)),
    (r"душанба\w*\s+(?:дан\s+)?шанба\w*", (0, 5)),
    (r"dushanba\w*\s+(?:dan\s+)?juma\w*", (0, 4)),
    (r"понедельник\w*\s+(?:по|до)\s+суббот\w*", (0, 5)),
    (r"понедельник\w*\s+(?:по|до)\s+пятниц\w*", (0, 4)),
    (r"\bmon\w*\s*[-–—to]+\s*sat\w*", (0, 5)),
    (r"\bmon\w*\s*[-–—to]+\s*fri\w*", (0, 4)),
]


def days_named_in(text: str | None) -> frozenset[int] | None:
    """The working week stated inside a free-form hours sentence, if it is.

    The clinic put its days and its hours in one variable, which is how a
    person would write it. Reading the days back out of it means the
    deployment does not have to repeat itself in a second variable -- and,
    more to the point, means the days come from something the clinic actually
    wrote rather than from a default in this file.
    """
    if not text:
        return None
    lowered = text.lower()
    for pattern, (first, last) in _WEEK_SPANS:
        if re.search(pattern, lowered):
            return frozenset(range(first, last + 1))
    return None


_DAY_NAMES: dict[str, int] = {
    "mon": 0, "dushanba": 0, "du": 0,
    "tue": 1, "seshanba": 1, "se": 1,
    "wed": 2, "chorshanba": 2, "cho": 2,
    "thu": 3, "payshanba": 3, "pay": 3,
    "fri": 4, "juma": 4, "ju": 4,
    "sat": 5, "shanba": 5, "sha": 5,
    "sun": 6, "yakshanba": 6, "yak": 6,
}


@dataclass(frozen=True)
class Schedule:
    """The clinic's opening rules, as configured."""

    days: frozenset[int]
    opens: time
    closes: time
    horizon_days: int

    def is_open_on(self, day: date) -> bool:
        return day.weekday() in self.days

    def is_open_at(self, at: time) -> bool:
        return self.opens <= at < self.closes


def _parse_hours(raw: str | None) -> tuple[time, time]:
    if not raw:
        raise ScheduleNotConfiguredError(
            "CLINIC_WORK_HOURS is not set. The clinic's opening hours are "
            "unknown, so the assistant cannot tell a patient when to come. "
            'Set it to something like "09:00 - 18:00".'
        )
    match = _HOURS.search(raw)
    if match is None:
        raise ScheduleNotConfiguredError(
            f"CLINIC_WORK_HOURS={raw!r} is not a pair of times. "
            'Expected something like "09:00 - 18:00".'
        )
    opens = time(int(match.group(1)), int(match.group(2)))
    closes = time(int(match.group(3)), int(match.group(4)))
    if closes <= opens:
        raise ScheduleNotConfiguredError(
            f"CLINIC_WORK_HOURS={raw!r} closes at or before it opens."
        )
    return opens, closes


def _parse_days(raw: str | None) -> frozenset[int]:
    if not raw:
        raise ScheduleNotConfiguredError(
            "CLINIC_WORK_DAYS is not set. Which days the clinic opens is "
            "unknown, so the assistant cannot refuse a day it is shut -- "
            'which is how a Sunday booking happened. Set it to e.g. "mon-sat".'
        )
    lowered = raw.strip().lower()
    days: set[int] = set()

    span = re.fullmatch(r"([a-z]+)\s*-\s*([a-z]+)", lowered)
    if span and span.group(1) in _DAY_NAMES and span.group(2) in _DAY_NAMES:
        first, last = _DAY_NAMES[span.group(1)], _DAY_NAMES[span.group(2)]
        current = first
        while True:
            days.add(current)
            if current == last:
                break
            current = (current + 1) % 7
        return frozenset(days)

    for part in re.split(r"[,\s]+", lowered):
        if not part:
            continue
        if part not in _DAY_NAMES:
            raise ScheduleNotConfiguredError(
                f"CLINIC_WORK_DAYS={raw!r} contains {part!r}, which is not a day."
            )
        days.add(_DAY_NAMES[part])
    if not days:
        raise ScheduleNotConfiguredError(f"CLINIC_WORK_DAYS={raw!r} names no days.")
    return frozenset(days)


def _parse_horizon(raw: int | None) -> int:
    """How far ahead a booking may be taken.

    Unlike the days and the hours, this falls back rather than refusing, and
    the asymmetry is deliberate. Being wrong about the horizon costs a patient
    one message -- they are told a date is too far off, and the front desk
    settles it. Being wrong about the *days* costs them a journey to a closed
    clinic. So the harsher rule guards the harsher failure.

    Refusing here would be worse than falling back for a second reason: it
    would take the whole schedule down with it, and with it the closed-day
    check. A deployment with correct hours and no horizon would go back to
    offering Sundays, which is the exact failure this module exists to stop.

    The fallback is not a guess. DEFAULT_SEARCH_HORIZON_DAYS is the value this
    system already searches the diary with, so using it here makes the two
    agree instead of introducing a third number.
    """
    if raw is None:
        logger.warning(
            "booking_horizon_not_configured using=%s "
            "detail=BOOKING_HORIZON_DAYS is unset; falling back to the value "
            "the schedule search already uses. Set it to the clinic's real "
            "booking window.",
            DEFAULT_SEARCH_HORIZON_DAYS,
        )
        return DEFAULT_SEARCH_HORIZON_DAYS
    if raw < 1:
        raise ScheduleNotConfiguredError(f"BOOKING_HORIZON_DAYS={raw} must be at least 1.")
    return raw


def load(settings: Settings | None = None) -> Schedule:
    """The configured schedule, or `ScheduleNotConfiguredError` naming what is missing."""
    resolved = settings or get_settings()
    opens, closes = _parse_hours(resolved.clinic_work_hours)
    # CLINIC_WORK_DAYS wins when it is set, because a deployment that states
    # the days separately means it. Otherwise the days are read out of the
    # hours sentence, which is where this clinic put them.
    configured_days = getattr(resolved, "clinic_work_days", None)
    days = (
        _parse_days(configured_days)
        if configured_days
        else days_named_in(resolved.clinic_work_hours)
    )
    if days is None:
        raise ScheduleNotConfiguredError(
            "Which days the clinic opens is not stated. Either set "
            "CLINIC_WORK_DAYS (e.g. \"mon-sat\"), or name the working week "
            "inside CLINIC_WORK_HOURS the way a patient would read it "
            '("Dushanbadan shanbagacha 09:00 dan 18:00 gacha").'
        )
    return Schedule(
        days=days,
        opens=opens,
        closes=closes,
        horizon_days=_parse_horizon(getattr(resolved, "booking_horizon_days", None)),
    )


def load_or_none(settings: Settings | None = None) -> Schedule | None:
    """The schedule, or None with the reason logged.

    For the paths that must keep working while the clinic is asked: the
    assistant goes on answering questions it can answer, and simply stops
    claiming anything about days and times.
    """
    try:
        return load(settings)
    except ScheduleNotConfiguredError as exc:
        # ERROR, not WARNING: a deployment in this state is telling patients
        # less than it should, and somebody has to notice.
        logger.error("clinic_schedule_not_configured detail=%s", exc)
        return None
