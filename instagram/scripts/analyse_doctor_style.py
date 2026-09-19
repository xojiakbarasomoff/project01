r"""Read the doctor's exported DMs and describe how the doctor actually writes.

The export is flat: every record is system/user/assistant, and the assistant
side is whatever the clinic's Instagram account sent. That is three different
authors wearing one label, and telling them apart is the whole job of this
script, because only one of them is worth imitating.

  * **Macros.** One 300-character block about a procedure, with "!!!" and a
    price, appears 1008 times verbatim. Nobody types that 1008 times: it is a
    saved reply, pasted by whoever was on the account. Long *and* repeated is
    the signature, and it is reliable -- the doctor's own repeated lines are
    short ones like "Yoshiz nechida".
  * **Platform artefacts.** "Click for audio" is Instagram telling us a voice
    note was sent. It is not a message at all.
  * **The doctor.** What is left: short, lowercase, misspelt, dialectal,
    asking one thing at a time. This is the voice the assistant should learn.

Macros are excluded from the style corpus rather than merely deprioritised.
They are exactly where the prices, addresses and phone numbers live, and
those are the facts the clinic said must come from the live system instead.

Nothing here writes to the database or to the prompt. It prints a report and,
with --write, saves the doctor-only corpus for app.services.style to load.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Long *and* seen this often means a saved reply, not a person typing.
# Measured, not guessed: in the clinic's export the doctor's most repeated
# lines are greetings of under 40 characters, while every text above this
# length that recurs at all turns out to be one of six pasted blocks.
MACRO_MIN_LENGTH = 120
MACRO_MIN_REPEATS = 5

_ARTEFACT = re.compile(
    r"click for audio|sent an attachment|liked a message|sent a voice"
    r"|audio call|missed a call|started a call|reacted\s",
    re.I,
)

# A reply that carries a fact which has since moved or changed: a price, a
# phone number, a street address. Separated from the doctor's voice however
# short it is, because "800$" is four characters and still the one thing the
# assistant must never learn from an old chat -- prices, hours and addresses
# come from the clinic's live records or not at all.
_CARRIES_A_FACT = re.compile(
    r"\d+\s*\$|\$\s*\d+|\d+\s*(?:ming|минг|mln|million|млн)\b"
    r"|\d[\d\s\-()]{6,}\d"
    r"|narx|нарх|цен[аыу]\b"
    r"|tuman|туман|kvartal|квартал|массив"
    r"|klinikasi\s*(?:da|ga)|клиникаси|medwell|plan\s*baby|marmed",
    re.I,
)


def _cyrillic(text: str) -> int:
    return len(re.findall(r"[Ѐ-ӿ]", text))


def _latin(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def script_of(text: str) -> str:
    cyrillic, latin = _cyrillic(text), _latin(text)
    if cyrillic == 0 and latin == 0:
        return "other"
    return "cyrillic" if cyrillic > latin else "latin"


def load(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            pairs.append((messages[1]["content"].strip(), messages[2]["content"].strip()))
    return pairs


def split_authors(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Sort every reply into artefact, macro or the doctor's own words."""
    repeats = Counter(reply for _, reply in pairs)
    buckets: dict[str, list[tuple[str, str]]] = {
        "artefact": [],
        "macro": [],
        "facts": [],
        "doctor": [],
    }
    for question, reply in pairs:
        if _ARTEFACT.search(reply) or _ARTEFACT.search(question):
            buckets["artefact"].append((question, reply))
        elif _CARRIES_A_FACT.search(reply):
            buckets["facts"].append((question, reply))
        elif len(reply) >= MACRO_MIN_LENGTH and repeats[reply] >= MACRO_MIN_REPEATS:
            buckets["macro"].append((question, reply))
        else:
            buckets["doctor"].append((question, reply))
    return buckets


# The things the report needs to be able to count, each one a question the
# style rules will have to answer.
_ADDRESS = re.compile(r"\b(aka|uka|opa|singlim|akajon|ака|ука|опа)\b", re.I)
_GREETING = re.compile(r"^\s*(?:va+lek|va+lay|vaalek|assalom|волек|ваалек|ассалом)", re.I)
_INVITE = re.compile(r"qabulga\s+kel|қабулга\s+кел|кабулга\s+кел|приходите", re.I)
_ASKS_BACK = re.compile(
    r"\?|\bnecha\b|\bnechida\b|\bqancha\b|\bqanaqa\b|\bqaysi\b|\bqayer|\bnima\b|неч",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<!\d)[.!?]+(?!\d)")


def describe(pairs: list[tuple[str, str]]) -> None:
    replies = [reply for _, reply in pairs]
    lengths = sorted(len(reply) for reply in replies)
    words = sorted(len(reply.split()) for reply in replies)

    def pct(matching: int) -> str:
        return f"{matching / len(replies):.0%}"

    print(f"  javoblar soni      {len(replies)}")
    print(f"  uzunlik (belgi)    median {lengths[len(lengths) // 2]}"
          f", p90 {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    print(f"  uzunlik (so'z)     median {words[len(words) // 2]}"
          f", p90 {words[int(len(words) * 0.9)]}")
    print(f"  nuqta/so'roq yo'q  {pct(sum(1 for r in replies if not _SENTENCE_END.search(r)))}")
    one_sentence = sum(1 for r in replies if len(_SENTENCE_END.findall(r)) <= 1)
    print(f"  1 gapdan iborat    {pct(one_sentence)}")
    print(f"  savol bilan javob  {pct(sum(1 for r in replies if _ASKS_BACK.search(r)))}")
    print(f"  'aka/uka/opa' deb  {pct(sum(1 for r in replies if _ADDRESS.search(r)))}")
    print(f"  salom bilan ochadi {pct(sum(1 for r in replies if _GREETING.match(r)))}")
    print(f"  qabulga taklif     {pct(sum(1 for r in replies if _INVITE.search(r)))}")
    scripts = Counter(script_of(reply) for reply in replies)
    total = sum(scripts.values())
    shares = ", ".join(
        f"{name} {count / total:.0%}" for name, count in scripts.most_common()
    )
    print("  alifbo             " + shares)
    matched = sum(1 for question, reply in pairs if script_of(question) == script_of(reply))
    print(f"  bemor alifbosiga mos {matched / len(pairs):.0%}")


# Positive evidence that a reply was written by the doctor or by the clinic's
# administrator. The earlier version of this script kept "whatever was left
# after the macros" -- which is not evidence of anything. The export carries
# no author field, so authorship has to be argued for, and a corpus of 2,300
# replies that provably came from the clinic is worth more than 6,500 that
# might have.
_STAFF_VOICE = re.compile(
    r"yordamchi|йордамчи"          # "I am the doctor's assistant"
    r"|eshitaman|эшитаман"          # "I'm listening"
    r"|qabulga\s+kel|қабулга\s+кел|кабулга\s+кел|korikka\s+kel"
    r"|yoshiz\s+nechida|necha\s+yoshsiz|йошиз\s+нечида"
    r"|qayerdans|қаерданс|каерданс"
    r"|shikoyati|шикоят|qanday\s+yordam|қандай\s+йордам"
    r"|vaalaykum|voaleykum|volekum|волекум|валекум|ваалекум"
    r"|assalomalekum|ассаломалекум",
    re.I,
)

# A patient's voice appearing where the clinic's should be. Whoever wrote it,
# it is not a reply the assistant should imitate.
_PATIENT_VOICE = re.compile(
    r"\bmenda\b|\bменда\b|\bmening\b|og'?riyap|огрияп"
    r"|yordam\s+bering|ёрдам\s+беринг"
    r"|\bмне\b|у\s+меня|nima\s+qilsam|нима\s+килсам",
    re.I,
)


def looks_like_the_clinic(reply: str) -> bool:
    """Whether this reply *resembles* the clinic's side of a conversation.

    Resembles, not "was written by". The export carries `role` and `content`
    and nothing else -- no sender id, no author, no timestamp. Every reply is
    labelled `assistant` because every reply went out from the clinic's one
    Instagram account, and that account was used by the doctor, by an
    administrator, by whoever pasted the saved replies, and possibly by an
    earlier bot. Which of them typed any given line is not recoverable from
    this file, and no regular expression makes it recoverable.

    So this is a shortlist, not a verdict. What it produces goes to
    data/clinic_voice_candidates.json for a person at the clinic to approve;
    what the assistant actually shows the model is the approved list, which
    starts empty.
    """
    return bool(_STAFF_VOICE.search(reply)) and not _PATIENT_VOICE.search(reply)


# What must not survive into a file that sits in the repository. The patient
# side of these pairs is somebody's real message: their number, their name,
# what is wrong with them. The style corpus needs the shape of the question,
# never the person who asked it.
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\d+][\d\s\-()+.]{5,}\d"), "[raqam]"),
    (re.compile(r"https?://\S+|instagram\.com/\S+"), "[havola]"),
    # Instagram handles, which survive in more shapes than the obvious one.
    # "XUDOYOR_88_08" is somebody's account: upper case, single underscores,
    # no "@". It reached the file because the first version of this rule only
    # looked for lower case and doubled underscores -- a test comparing the
    # corpus against the redaction rules is what found it.
    (re.compile(r"@[A-Za-z0-9._]{3,}", re.I), "[profil]"),
    (re.compile(r"\b[A-Za-z][A-Za-z0-9.]*(?:_[A-Za-z0-9.]+)+\b"), "[profil]"),
    (
        re.compile(
            r"(ism(?:im)?|ot(?:im)?|исм(?:им)?|меня\s+зовут|зовут)"
            r"\s*[-:]?\s*[A-Za-zЀ-ӿ']{3,30}",
            re.I,
        ),
        r"\1 [ism]",
    ),
    # No leading word boundary: the export runs words together often enough
    # ("ketadim8 Yosh") that requiring one leaves ages in the file.
    (re.compile(r"\d{1,2}\s*(?:yosh\w*|ёш\w*|лет|года)\b", re.I), "[yosh]"),
]


def redact(text: str) -> str:
    """Strip anything that identifies the person who wrote this message."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the exported .jsonl")
    parser.add_argument("--write", type=Path, default=None, help="save the doctor-only corpus here")
    args = parser.parse_args()

    pairs = load(args.source)
    buckets = split_authors(pairs)

    print(f"jami yozuv: {len(pairs)}\n")
    for name, label in (
        ("artefact", "platforma izi (Click for audio va h.k.)"),
        ("macro", "shablon/makro javob (tayyor matn)"),
        ("facts", "o'zgaruvchan fakt (narx/telefon/manzil)"),
        ("doctor", "doktorning o'z so'zlari"),
    ):
        print(f"{label}: {len(buckets[name])} ({len(buckets[name]) / len(pairs):.0%})")
    print()

    print("=== DOKTORNING USLUBI ===")
    describe(buckets["doctor"])

    print("\n=== SHABLON JAVOBLAR (uslub namunasi sifatida ISHLATILMAYDI) ===")
    macros = Counter(reply for _, reply in buckets["macro"])
    for text, count in macros.most_common(8):
        print(f"  {count:5}x  {text[:90]}")

    candidates = [(q, a) for q, a in buckets["doctor"] if looks_like_the_clinic(a)]
    print(f"\nnomzodlar: {len(candidates)} / {len(buckets['doctor'])}")

    if args.write is not None:
        args.write.write_text(
            json.dumps(
                [
                    {"id": index, "patient": redact(question), "clinic": redact(reply)}
                    for index, (question, reply) in enumerate(candidates)
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"yozildi: {args.write} ({len(candidates)} nomzod, shaxsiy ma'lumot tozalangan)")
        print("DIQQAT: bular TASDIQLANMAGAN. Eksportda muallif maydoni yo'q,")
        print("shuning uchun ularni 'doktor yozgan' deb atash mumkin emas.")
        print("Promptga faqat data/clinic_voice_approved.json tushadi.")


if __name__ == "__main__":
    main()
