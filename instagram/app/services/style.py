"""The clinic's own voice, shown to the model as examples rather than described.

Telling a model "reply briefly and warmly" produces a model's idea of brief
and warm. Showing it what this clinic's administrator actually wrote produces
something closer to the clinic. So a handful of real exchanges go into the
prompt, chosen to resemble the message being answered.

**What these examples are and are not.** They are tone: sentence length,
directness, how an invitation to come in is phrased. They are not facts and
they are not memory. A patient's name, number, language and the state of
their request are read from `users` and `conversation_states` on every turn
(app.services.turn); nothing about the person being answered comes from an
old chat with somebody else. The examples are stripped of identifying detail
before they ever reach this file -- see scripts/analyse_doctor_style.py.

**Nothing is used until a person at the clinic approves it, and that is not
caution, it is the only honest option.** The export carries `role` and
`content` and nothing else -- no sender id, no author, no timestamp. Every
reply is labelled `assistant` because every reply left the clinic's one
Instagram account, which was used by the doctor, by an administrator, by
whoever pasted the saved replies and possibly by an earlier bot. Which human
typed any given line is not in the file and cannot be inferred from it.

An earlier version of this called its 2,015 shortlisted pairs "written by the
doctor". They were not shown to be. A heuristic that keeps replies containing
"eshitaman" or "qabulga keling" produces a plausible shortlist and no more
than that, and a bot taught to imitate a line the patient actually wrote is a
bot that learns to sound like a patient.

So there are two files:

  data/clinic_voice_candidates.json   2,015 shortlisted pairs, for review.
  data/clinic_voice_approved.json     what a person has signed off. Empty.

Only the second is read here. Until the clinic works through the first, this
module contributes nothing to the prompt, and the assistant writes as it did
before -- which is the correct behaviour for a feature whose input has not
been verified.
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# What a person at the clinic has approved. Starts as an empty list, and an
# empty list means no style block in the prompt.
CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "clinic_voice_approved.json"

# The shortlist waiting for that review. Read by nothing on the reply path;
# it exists so there is something to review.
CANDIDATES_PATH = Path(__file__).resolve().parents[2] / "data" / "clinic_voice_candidates.json"

# How many examples go in front of the model. Few, on purpose: they are there
# to set a tone, and a dozen of them would crowd out the retrieved FAQ that
# answers what the patient actually asked.
EXAMPLES_IN_PROMPT = 4


@dataclass(frozen=True)
class Example:
    patient: str
    clinic: str


@lru_cache(maxsize=1)
def corpus() -> tuple[Example, ...]:
    """Every usable example, read once.

    A missing or unreadable file is not fatal. The assistant answers without
    examples exactly as it did before there were any; losing the tone is
    worth less than refusing to reply.
    """
    try:
        raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("style_corpus_unavailable path=%s", CORPUS_PATH)
        return ()
    return tuple(
        Example(patient=row["patient"], clinic=row["clinic"])
        for row in raw
        if row.get("patient") and row.get("clinic")
    )


_WORD = re.compile(r"[\w']+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if len(word) > 3}


def choose(
    message: str, *, limit: int = EXAMPLES_IN_PROMPT, seed: int | None = None
) -> list[Example]:
    """Examples resembling `message`, by shared words.

    Overlap rather than embeddings: this picks writing to imitate, not facts
    to rely on, so an approximate match costs nothing and an embedding call
    per turn would cost real time on the path that answers a patient.

    When nothing overlaps -- a greeting, a message in an alphabet the corpus
    is thin on -- a stable sample is returned rather than none, because a
    greeting is exactly where the clinic's voice is most recognisable.
    """
    pool = corpus()
    if not pool:
        return []
    wanted = _words(message)
    if wanted:
        scored = sorted(
            pool,
            key=lambda example: len(wanted & _words(example.patient)),
            reverse=True,
        )
        best = [example for example in scored if wanted & _words(example.patient)][:limit]
        if best:
            return best
    return random.Random(seed if seed is not None else 0).sample(
        list(pool), min(limit, len(pool))
    )


def render(examples: list[Example]) -> str:
    """The examples as a prompt block, labelled so they cannot be mistaken for facts."""
    if not examples:
        return ""
    lines = [
        "\n\nHOW THIS CLINIC WRITES",
        "Exchanges from this clinic's own inbox that clinic staff have approved,",
        "with patient details removed.",
        "Copy the tone -- short, direct, one question at a time, warm without",
        "being formal. Do NOT copy any fact, price, time or spelling mistake",
        "from them, and do not treat them as anything this patient said.",
    ]
    for example in examples:
        lines.append(f"  Patient: {example.patient}")
        lines.append(f"  Clinic:  {example.clinic}")
    return "\n".join(lines) + "\n"
