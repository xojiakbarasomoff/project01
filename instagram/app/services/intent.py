"""What the patient is asking for, decided before anything is generated.

There was no such decision before. Every message went guardrail -> retrieval
-> model, and the guardrail's only question was "does this sound medical?".
So "Qabulga yozilmoqchiman, prostatada og'riq bor" -- a booking request with
a symptom in it, which is what almost every booking request looks like --
was classified as a medical question and answered with a medical deflection.
The patient asked to come in and was told the assistant cannot give advice.

Rules, not a model call. The clinic's messages are short and the vocabulary
is small, so rules get this right nearly always, cost nothing, run in
microseconds and can be tested -- and when they cannot place a message they
say UNKNOWN, and it is handled as ordinary conversation, which is the safe
direction to be wrong in.

Order matters, and the first rule is the one that fixes the complaint: a
message that asks to be seen is a booking request *whatever else it also
contains*. Symptoms are why people book.
"""

import re
from enum import StrEnum

from app.models.conversation_state import ConversationState, FlowStatus
from app.services import when as when_service


class Intent(StrEnum):
    GREETING = "greeting"
    BOOKING_REQUEST = "booking_request"
    BOOKING_DATE = "booking_date"
    BOOKING_TIME = "booking_time"
    BOOKING_CONFIRM = "booking_confirm"
    RESCHEDULE = "reschedule"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_CONFIRM = "cancel_confirm"
    EXISTING_BOOKING_QUERY = "existing_booking_query"
    PRICE_QUESTION = "price_question"
    LOCATION_QUESTION = "location_question"
    HOURS_QUESTION = "hours_question"
    DOCTOR_QUESTION = "doctor_question"
    MEDICAL_QUESTION = "medical_question"
    THANKS = "thanks"
    INJECTION_ATTEMPT = "injection_attempt"
    UNKNOWN = "unknown"


# Asking to be seen. Checked before anything medical, which is the whole
# point: "yozilmoqchiman" outranks "og'riyapti" in the same sentence.
_WANTS_AN_APPOINTMENT = re.compile(
    r"qabulga\s+yoz|qabulga\s+kel|yozilmoqchi|yozilsam|yozib\s+qo|navbat\s+ol|navbatga"
    r"|band\s+qil|kelsam\s+bo|uchrash"
    r"|қабулга\s+ёз|навбат|ёзилмоқчи"
    r"|записаться|запишите|на\s+приём|на\s+прием",
    re.IGNORECASE,
)
_CANCEL = re.compile(
    r"bekor\s*qil|бекор\s*қил|отмен(?:ить|и|яю)|аннулир",
    re.IGNORECASE,
)
_RESCHEDULE = re.compile(
    r"o'?zgartir|ko'?chir|boshqa\s+(?:kun|vaqt)ga|surib|перенес|поменя",
    re.IGNORECASE,
)
_EXISTING = re.compile(
    r"(?:qabul\w*|navbat\w*)[^.?!]{0,30}(?:bormi|qachon|nechida|qaysi)"
    r"|yozilganmanmi|qabulim\s+bor|моя\s+запис|когда\s+у\s+меня",
    re.IGNORECASE,
)

_PRICE = re.compile(
    r"narx|qancha\s+tur|qanchaga|pul|to'?lov|нарх|қанча"
    r"|сколько\s+сто|цен|оплат",
    re.IGNORECASE,
)
_LOCATION = re.compile(
    r"manzil|qayerda|qayerdasiz|mo'?ljal|манзил|қаерда"
    r"|адрес|где\s+наход|как\s+доехать",
    re.IGNORECASE,
)
_HOURS = re.compile(
    r"ish\s*vaqt|nechigacha|nechidan|soat\s+nechada|ochiq|yopiq|dam\s+olish\s+kuni"
    r"|иш\s*вақт|график|во\s+сколько|работаете",
    re.IGNORECASE,
)
_DOCTOR = re.compile(
    r"shifokor|doktor|vrach|ayol\s+shifokor|tajriba|шифокор|духтир|врач|стаж",
    re.IGNORECASE,
)
_MEDICAL = re.compile(
    r"og'?ri|ogri|shish|qichi|qon\s+kel|siydik|buyrak|prostat|jinsiy|erek|bepusht"
    r"|analiz|tahlil|\bpsa\b|gormon|infeksiya|muammo\s+bor|kasal|davola"
    r"|оғри|буйрак|простат|анализ|болит|проблем|леч",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^\s*(?:as+alom\w*|salom\w*|vaalaykum|xayrli|hayrli|hello|hi"
    r"|ассалом\w*|салом\w*|здравствуйте|привет)",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"^\s*(?:katta\s+)?(?:rahmat|raxmat|tashakkur|спасибо|рахмат|thanks)[\s!.,)]*$",
    re.IGNORECASE,
)

# Somebody reading for the machinery rather than talking to the clinic. Not
# an error and not a refusal: the caller answers it as an ordinary patient
# message and simply does not act on it.
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?previous|system\s*prompt|internal\s+(?:rules?|instruction)"
    r"|developer\s+(?:mode|message)|jailbreak|prompt\s*injection"
    r"|qoidalar\w*\s+(?:nima|qanday|ko'?rsat|ayt)|ichki\s+(?:qoida|prompt|ko'?rsatma)"
    r"|системн\w*\s+промпт|внутренн\w+\s+правил|покажи\s+правил"
    r"|\[\[|\bBOOK:|\bCALLBACK:",
    re.IGNORECASE,
)

_YES = re.compile(
    r"^\s*(?:ha|xa|ha'?a|shu|shunga|mayli|bo'?ladi|to'?g'?ri|tasdiqla\w*"
    r"|да|хорошо|давай|ок|ok|okay)[\s!.,)]*$",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^\s*(?:yo'?q|yoq|kerak\s+emas|нет|не\s+надо)[\s!.,]*$",
    re.IGNORECASE,
)


def classify(message: str, *, state: ConversationState | None = None) -> Intent:
    """The intent of one message, read in the light of where the flow is.

    The state comes first for exactly one class of message: the short ones.
    "Ha", "shu" and "12" mean nothing on their own and everything when
    something is waiting for them, which is why they are interpreted against
    `state.status` rather than against a dictionary.
    """
    text = message.strip()
    if not text:
        return Intent.UNKNOWN

    if _INJECTION.search(text):
        return Intent.INJECTION_ATTEMPT

    status = FlowStatus(state.status) if state is not None else FlowStatus.IDLE

    # 1. An answer to the question the assistant just asked.
    if status is FlowStatus.AWAITING_CANCEL_CONFIRM and _YES.match(text):
        return Intent.CANCEL_CONFIRM
    if status is FlowStatus.AWAITING_CONFIRMATION:
        if _YES.match(text):
            return Intent.BOOKING_CONFIRM
        if _NO.match(text):
            return Intent.BOOKING_REQUEST
    if status in {FlowStatus.COLLECTING, FlowStatus.AWAITING_DATE, FlowStatus.AWAITING_TIME}:
        found = when_service.read(text)
        if found.at is not None:
            return Intent.BOOKING_TIME
        if found.day is not None:
            return Intent.BOOKING_DATE

    # 2. What the patient plainly said. Booking outranks symptoms.
    if _CANCEL.search(text):
        return Intent.CANCEL_REQUEST
    if _RESCHEDULE.search(text):
        return Intent.RESCHEDULE
    if _EXISTING.search(text):
        return Intent.EXISTING_BOOKING_QUERY
    if _WANTS_AN_APPOINTMENT.search(text):
        return Intent.BOOKING_REQUEST
    if _THANKS.match(text):
        return Intent.THANKS

    # 3. A bare time with no flow open is still an answer to something.
    if when_service.read_time(text) is not None and len(text) <= 40:
        return Intent.BOOKING_TIME

    # 4. Subjects, in the order that decides what the reply is made of.
    #    Price, address and hours come before "medical" so that "prostat
    #    tekshiruvi qancha turadi" is a price question, which the clinic
    #    answers from its own records, and not a symptom to deflect.
    if _PRICE.search(text):
        return Intent.PRICE_QUESTION
    if _LOCATION.search(text):
        return Intent.LOCATION_QUESTION
    if _HOURS.search(text):
        return Intent.HOURS_QUESTION
    if _DOCTOR.search(text):
        return Intent.DOCTOR_QUESTION
    if _MEDICAL.search(text):
        return Intent.MEDICAL_QUESTION

    if when_service.read_day(text) is not None and len(text) <= 40:
        return Intent.BOOKING_DATE
    if _GREETING.match(text) and len(text) <= 40:
        return Intent.GREETING

    return Intent.UNKNOWN


# Which intents put a patient into the collection flow at all. Everything
# else must not start asking for a name and a number, which is the second
# half of the complaint this module exists to answer.
BOOKING_INTENTS = frozenset(
    {
        Intent.BOOKING_REQUEST,
        Intent.BOOKING_DATE,
        Intent.BOOKING_TIME,
        Intent.BOOKING_CONFIRM,
        Intent.RESCHEDULE,
    }
)
