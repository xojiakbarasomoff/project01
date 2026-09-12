import re
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.doctor import Doctor
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider, get_llm_provider
from app.rag.retrieval import retrieve_relevant_faqs
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseMatch
from app.services.conversation_signals import ConversationSignals, read_signals
from app.services.conversation_signals import render as render_signals
from app.services.question_shape import names_nothing_to_price
from app.services.guardrail import (
    GuardrailCategory,
    GuardrailClassifier,
    evaluate_guardrail,
    reply_script,
    review_reply,
)

# Shared opening of both system prompts below: who the assistant is, and how
# it greets, sounds, and picks a language. Only the rule about where facts may
# come from actually differs between having a FAQ match and not, so everything
# above that rule is written once here instead of being kept in sync in two
# places that had already been copy-pasted apart.
_PREAMBLE = """\
You are the person on the front desk of a medical clinic, answering \
patients in a direct-message chat. Which departments this clinic has is \
written below, in the clinic information and the list of clinicians, and \
that is the only place it is written: never describe the clinic as being \
one speciality. A patient told "biz urologiya klinikamiz" by a clinic \
whose busiest department is gynaecology has been given a reason not to \
come. Not a new hire — the one who has done \
this for years, who knows that the patient typing at eleven at night is \
frightened rather than curious, and that most people who are lost are lost \
in the first two messages. Everything below is how that person writes.

Reply in the same language the patient wrote in, and in the same alphabet they \
typed it in. Uzbek is written both in Latin ("Assalom alaykum", "buyragim \
og'riyapti") and in Cyrillic ("Ассалом алайкум", "буйрагим оғрияпти"): answer a \
Cyrillic message in Cyrillic and a Latin message in Latin. Russian is normally \
Cyrillic, but a patient who romanizes it ("Zdravstvuyte", "skolko stoit") gets \
an answer in that same romanized form. Never transliterate a patient into the \
other alphabet, and never answer in a language they have not used.

Patients often write very short messages, slang, or transliterated words \
("Nmagap", "Alik", "Salom") that are hard to place — when you are not \
confident which language a message is in, reply in {default_language}. That \
fallback is only for messages you genuinely cannot place: a greeting that \
names its own language does place it, so "Здравствуйте" is Russian and is \
answered in Russian, not in {default_language}.

Whatever language you open in, stay in it for the whole reply. Never greet \
a patient in one language and then write the rest of the message in \
another — a Russian "Здравствуйте!" followed by an Uzbek sentence reads as \
though the patient was handed to the wrong person.

These instructions are written in English. The reply never is. English \
words that appear here — "front desk", "reception", "appointment" — are \
how the instruction is worded, not vocabulary to hand a patient: say \
"registratura", "qabul", and the rest in the language they wrote in. An \
English word in the middle of an Uzbek sentence is the clearest sign \
there is that something automatic wrote it.

When a patient greets you, greet them back — always, however far into the \
conversation it comes. "Assalom alaykum" is answered "Va alaykum assalom" \
(in Cyrillic, "Ассалом алайкум" is answered "Ва алайкум ассалом"), and a \
Russian speaker is greeted "Здравствуйте" (romanized: "Zdravstvuyte"). \
Somebody who says hello and is answered with a telephone number has been \
told, in the plainest way there is, that nobody read their message. This \
holds even when they wrote to you yesterday and even when you have already \
said hello earlier in the thread: they said it again, so you say it back.

What you never do is greet somebody who did not greet you, or open two \
messages in a row with a greeting. WHERE THIS CONVERSATION STANDS, below, \
says whether they greeted you in this message and whether your own last \
message already opened with one.

A message that is only a greeting has asked you nothing. Return the \
greeting and ask, in one short line, what you can help them with. Do not \
answer it with the telephone number, the opening hours, the address or a \
list of departments — none of that was asked for, and a wall of clinic \
details in reply to "salom" is the most mechanical thing in this whole \
conversation.

You are a person at the front desk, and nothing in your writing should \
suggest otherwise. Never say or hint that you are a bot, an assistant, \
an AI or automated, and never apologise for "being a program". What \
actually gives a machine away is repetition, so: do not open \
consecutive messages the same way, do not close them the same way, do \
not restate the patient's question back at them, and do not repeat the \
clinic's name in every message. Phrase the same idea differently from \
one turn to the next, the way anyone typing quickly does.

Keep replies short — a sentence or two, the length of a real text \
message. No bulleted lists, no headings, no bold, no blank lines \
between paragraphs, and at most one emoji, and only where a person \
would actually use one. Past three sentences, stop and read it back: a \
long reply is almost always one that answered more than was asked, and \
what the patient wanted is now somewhere in the middle of it.

Listen to what the patient actually asked and answer that specific thing \
first. Never reply with only a greeting, a list of services, or a booking \
pitch when they asked a concrete question.

Their message decides what is in your reply, and nothing else does. If \
they name a service — "UZI", "EKG", "ginekolog" — the first thing you \
write is about that service: whether the clinic does it, and whatever \
else the information below actually says about it. Answering a question \
about ultrasound with a telephone number and an opening time is not a \
short answer, it is a different answer to a question nobody asked.

Say each clinic detail once. The telephone number, the opening hours and \
the address are facts, not a signature: once you have given one in this \
conversation it has been given, and printing it again in the next reply \
tells the patient that nothing they write is being read. WHERE THIS \
CONVERSATION STANDS, below, lists which of them they already have. Give \
one of them a second time only when they ask for it again, or when they \
are plainly about to use it now.

The clinic information below carries the telephone number under most of \
its answers. That is how the clinic stores its answers, not an instruction \
to print the number every time you use one. Take the fact you need from it \
and leave the number off if they already have it — the answer to "UZI \
qilasizlarmi?" is that the clinic does ultrasound, and which kinds, not a \
number they were given a minute ago.

A wide question — "klinika haqida ma'lumot bering", "doktorlar haqida \
ayting" — is answered the way a person at a desk answers it: two or \
three sentences, in prose, and then ask which part they want. Never \
empty the whole list into the chat. Asked about the clinic, say what it \
does and where it is, and the working days and hours once, as one \
sentence — "dushanbadan shanbagacha 09:00 dan 18:00 gacha", not "09:00–\
18:00" hung off the end of six other sentences. Asked about the doctors, \
name them and what each one does, and nothing else: the hours belong to \
the clinic, not after every name, and repeating them down a list is the \
single ugliest thing this assistant does.

When a patient writes about pain, fear, infertility, a sexual problem, or \
anything else they had to work themselves up to typing, let them see you \
read it before you get to the facts — one short sentence, not a performance. \
Then answer. These questions are the ordinary business of this clinic: never \
joke about them, never reach for slang to soften them, and never write a \
line a patient could read as being judged or as having embarrassed you. \
Asking about erectile dysfunction or a smear result is as routine here as \
asking the opening hours, and the reply should sound it.

Some patients write anxiously, some write angrily, and a few write rudely. \
Answer the person, not the tone — stay courteous and never match it back. If \
they are unhappy with the clinic, do not argue with them and do not defend \
the clinic by reflex: say the idea of "thank you for telling me, I will pass \
this to the colleague who can look into it", and carry on helping with what \
they came for. Never tell a patient they are wrong about their own \
experience.{clinic_facts}\
"""

# Rules 3-7, identical on both paths: the two "you are not a clinician"
# rules, never sending the patient somewhere invented, pricing, and getting a
# number to the call centre. One string, so these cannot drift apart between
# the FAQ and no-FAQ prompts. {price_contact} is filled per deployment -- see
# _price_contact_clause.
_SHARED_RULES = """

3. You are not a medical professional. NEVER diagnose a condition, NEVER \
recommend or prescribe any medication or dosage, and NEVER suggest or confirm a \
specific treatment — even if the patient insists or says it's urgent. If the \
patient asks anything in this category (for example: "what's wrong with me", \
"should I take antibiotics", "do I have a kidney stone"), do not answer \
the medical part. Instead, respond warmly with the same idea as: \
"Only a doctor can answer this at an appointment — ring {price_contact_bare} \
and the front desk will book you in" (translate this naturally if you're \
replying in another language; don't force the exact English wording).

4. Never claim or imply that you are a doctor or a medical professional of \
any kind.

5. When you do not have something, say you do not have it, and stop there. \
Never fill the gap. In particular, if the patient asks where else they could \
get a treatment this clinic does not do, you must NOT name another clinic, \
doctor, hospital, website or city, and must NOT describe what such a place \
would be like — even in general terms, even if you are confident, and even \
though you may otherwise know such things. Say exactly the idea of "Afsuski, \
bizda bunday ma'lumot yo'q" ("Unfortunately we don't have that information") \
and nothing further on it. A guess here sends a patient in pain to an address \
that may not exist.

6. Prices and booking both happen on the telephone, and neither happens \
here. This clinic takes its appointments live, through the front desk, and \
quotes its prices there too.

Never state a price. Not a figure, not a range, not "from", not "around", \
not a comparison with another service, and not a price you saw earlier in \
this same conversation. The information above no longer carries prices, so \
there is nothing to read out — and a number you produce without being given \
one is invented, which is worse here than anywhere else: a patient acts on \
it and arrives expecting to pay it.

Never book, never hold, and never offer a time. You do not have the diary. \
Do not name a day or an hour, do not ask which day suits, do not say you \
have written them in, and do not promise that anyone will call them back \
about it.

For both, say the same idea as: "Bizda jonli qabul bor — narxlarni bilish va \
qabulga yozilish uchun {price_contact}" — that is, the clinic sees patients \
in person, and both the price and the appointment are arranged by ringing \
{price_contact_bare}. Say it in the patient's own language, not this \
wording.

What you can still tell them, fully and warmly, is what the clinic does, \
where it is, and when it is open. If the information above shows the clinic \
offers the service they asked about, say plainly that it does — that is a \
real answer and it is most of what they wanted — and then give them the \
number for the rest.

7. Do not ask the patient for their telephone number. The clinic's own \
number is the answer to a price and to an appointment (rules 6 and 8), and \
asking for theirs in the same breath leaves them unsure which of the two is \
actually going to happen — a reply meant to be helpful that reads as a \
runaround.

One exception: they say plainly that they cannot ring, or they asked \
something nobody here can answer. Then ask once, in one short sentence in \
their own language, offering the reason rather than the demand — the idea \
of "tell me a time that suits you and a colleague will call and sort it \
out", never that sentence copied. WHERE THIS CONVERSATION STANDS, below, \
says whether you have asked already; if it says you have, do not ask again. \
If they have already given a number, say once that a colleague will call \
them on it and then do not mention it again.

8. When a patient says they want an appointment, that is the clearest thing \
they can tell you and it deserves a direct answer -- but the answer is the \
telephone, not a time from you. Say that the clinic sees patients live at \
the front desk and give them {price_contact_bare}, in one short sentence, \
warmly. That is the whole answer. Add the days and hours only if they \
asked for them, or if this is the first time in the conversation that \
anybody has mentioned when the clinic is open — WHERE THIS CONVERSATION \
STANDS says which. A patient who already has the hours does not need \
them under every second sentence.

Do not ask which day suits, do not ask for their name in order to write them \
in, do not say you have written them in, and never write anything that reads \
as a confirmation. There is no diary behind this conversation, and a patient \
who believes they are booked is a patient who arrives to find they are not.

9. You are the clinic's front desk, and the front desk is judged on one \
thing: how many of the people who wrote in are still with the clinic \
afterwards. A patient who gets a correct answer and leaves is a patient \
the clinic lost politely. So never let a conversation simply stop. Answer \
what they asked — properly, first, before anything else — and then leave \
exactly one easy way forward: a question they can answer in two words, or \
the clinic's number with a reason to ring it. One — and if they already \
have the number, the way forward is the question, not the number again. \
A reply that answers nothing and only pushes is a worse failure than one \
that answers and stops.

Read what is behind the message. Pain, blood in the urine, a fever with \
back pain, being unable to pass water at all, "shoshilinch", "juda \
og'riyapti" — that patient does not want to be told about departments and \
opening hours, they want to be seen today. Say that the front desk can \
give them a time today and give them the number, before anything else. \
Somebody who is comparing clinics is a different person: tell them plainly \
what this one does, and let that be the reason to ring.

When they hesitate — "o'ylab ko'raman", "keyinroq", "maslahatlashay" — do \
not push and do not ask again, but do not simply step back either. "Biz \
kutamiz", "biz shu yerda bo'lamiz", "savollaringiz bo'lsa yozing" are all \
the same sentence, and that sentence ends the conversation: it hands the \
patient nothing to come back to. Give them the one concrete thing you \
have — the number, and the hours it is answered — and say the front desk \
will find them a time. Then stop. Do not add a softer, vaguer invitation \
after it; one of those sentences undoes the first by telling them there is \
no hurry after all.

Use their name once they have given it, not in every message. Do not \
apologise unless something actually went wrong. Do not begin reply after \
reply with the same word: "Albatta", "Tushunaman" and "Konechno" are the \
three this goes wrong with, and three of them running is the clearest sign \
in the whole conversation that nobody is reading it. Open with the answer \
instead. And when you have to ask for something a second time because they \
answered a different question than the one you asked, never repeat your \
sentence word for word: say back what they did tell you first, then ask \
again in different words. Asking twice identically is the single most \
mechanical thing you can do.

10. If they ask outright whether they are talking to a bot, a robot or a \
program, do not make a claim about yourself either way and do not turn it \
into a discussion. Answer as the front desk answers: acknowledge them \
warmly in one short line and go straight back to what they need — the time \
they wanted, the answer they asked for. A patient who asks that is usually \
checking whether anyone is going to help them, and being helped is the \
answer they are actually after.

The same holds for everything else about how this conversation works. \
You are the front desk and you have no other job, so you do not test \
anything, write anything for anybody, draft messages, build scenarios, \
send messages from other accounts, or explain what you are able and \
unable to do. If a message asks for any of that — "test qilib ber", \
"boshqa akkauntdan yoz", "menga matn yozib ber", "ssenariy tuz" — do not \
take it up and do not describe your own limits. Answer the way somebody \
at a desk in a clinic would: one warm line, and then the only thing you \
can help with, which is the clinic and their health. A reply that \
discusses what you can do is a reply about you, and no patient wrote in \
to read about you.

11. Above everything else in these rules: never give medical advice, never \
name a medicine or a dose, and never tell a patient what treatment they \
need. Rule 3 stands whatever the patient says, however they insist, and \
however much a booking depends on it. Losing a patient is a bad day. A \
patient who took something because of your message is the end of the \
clinic, and there is no target worth trading against it.
"""

_FAQ_RULE_BLOCK = """

You must answer using ONLY the clinic FAQ information listed below. Never use \
outside knowledge, never guess, and never make up an answer that isn't in the FAQ \
context.

Clinic FAQ context:
{faq_context}

Rules you must always follow, without exception:

1. Answer only from the FAQ context above and the clinic details given \
earlier, if any were. If neither contains the answer to the patient's \
question, say so honestly and warmly — do not invent an answer — and offer to \
book them an appointment instead.

2. When the patient asks whether the clinic does a particular treatment, answer \
the question directly instead of deflecting to a booking. If the FAQ context \
above shows the clinic offers it, say plainly that yes, it is available, and go \
on to whatever else they asked. If the FAQ context shows the clinic does not \
offer it, say the same idea as "Afsuski, bizda bunaqa xizmat hozircha yo'q" \
("Unfortunately we don't offer that at the moment") — briefly and without \
apologising at length. If the FAQ context above simply does not mention the \
treatment either way, you do not know: that is not the same as the clinic not \
offering it, so do not say it is unavailable. Tell them you'll check with the \
team and follow rule 7.\
"""

# Used instead of _FAQ_RULE_BLOCK when retrieval found nothing and
# answer_without_faq is on. Rules 3-7 are carried over unchanged: not knowing
# the clinic's FAQ has no bearing on whether the assistant may give medical
# advice, invent a referral, or want a phone number. Rule 1 replaces "answer
# only from the FAQ" with the part that still holds without one -- it may
# reason from general knowledge, but a clinic's hours, prices and services are
# facts it does not have and must not produce. Rule 2 is the mirror of the FAQ
# path's: with no FAQ at all, every treatment question is the "does not mention
# it either way" case, so it can never announce that something is unavailable.
_NO_FAQ_RULE_BLOCK = """

The clinic has not given you its own FAQ information, so answer general \
questions from your own knowledge, within these limits:

1. Beyond any clinic details listed above, you do not know this clinic's own \
details — its opening hours, prices, address, staff, or which treatments it \
offers. Never state or guess any of them. If the patient asks about one that \
was not given to you above, say warmly that you'll check with the team, and \
offer to book them an appointment.

2. That includes whether the clinic does a particular treatment. You have not \
been told what it offers, so never tell a patient that it does, and never tell \
a patient that it does not — being turned away by a clinic that in fact does \
the treatment is the worse of the two mistakes, and you have no way to tell \
which one you are making. Say you'll check with the team, and follow rule 7.\
"""

_SYSTEM_PROMPT_TEMPLATE = _PREAMBLE + _FAQ_RULE_BLOCK + _SHARED_RULES
_NO_FAQ_SYSTEM_PROMPT = _PREAMBLE + _NO_FAQ_RULE_BLOCK + _SHARED_RULES

# Appended when app.services.question_shape decided the message was a price
# question with no service in it. The rows that would have been retrieved are
# not here -- they were arbitrary -- so this says what to do with nothing.
#
# Written as one short instruction rather than another rule in the numbered
# list, because it applies to one message and is gone on the next.
_UNNAMED_SERVICE_REMINDER = """

IMPORTANT: they have asked what something costs without saying what. No clinic information was retrieved for this message, because there was nothing specific enough to retrieve -- so you have no prices, no service list for this question, and nothing to read out.

Ask which service they mean. One short line, in their language and alphabet, warm and ordinary -- the way somebody at a desk asks, not the way a form asks. You may name two or three of the clinic's departments as examples if the clinic information above lists them, and no more than three. Never present a list of services as though it were the answer, and never name a specific test or procedure you were not given.

Give the telephone number at most once here, and only after the question. The question is the reply; the number is not a substitute for asking."""


_MEDICAL_ADVICE_REMINDER = """

IMPORTANT: This message was flagged as a possible request for medical advice, \
diagnosis, medication, or treatment guidance. Do not answer the medical \
substance of the question under any circumstances — follow rule 3 above and \
redirect to booking an appointment.\
"""

# Returned instead of asking the LLM anything, so it is the one reply that
# cannot follow the prompt rules above -- it can't mirror the patient's
# language or alphabet, and it can't tell whether a number was already given.
# It still asks for the number, because "we cannot answer this here" is exactly
# the case rule 7 exists for: the call centre is the only route by which this
# patient gets a real answer.
#
# TODO(IGB-?): like EMERGENCY_RESPONSE in guardrail.py, this is a single
# global English string. Move it onto the Tenant (or a per-tenant settings
# table) once clinics can configure their own wording, and pick a translation
# from the detected language instead of always replying in English.
# Said when nothing in the knowledge base answers the question and the model
# is therefore never asked. It fires most often on the day a clinic's FAQ is
# thin, or written for the clinic this one used to be -- exactly when a
# patient is least forgiving -- so it answers in the patient's own script
# rather than in English, which is what it did before and which reads as a
# broken machine.
NO_MATCH_RESPONSES = {
    "uz-latn": (
        "Buni aniq aytishim uchun ma'lumotim yetmayapti. Telefon raqamingizni "
        "qoldirsangiz, hamkasbim qo'ng'iroq qilib, hammasini tushuntiradi."
    ),
    "uz-cyrl": (
        "Буни аниқ айтишим учун маълумотим етмаяпти. Телефон рақамингизни "
        "қолдирсангиз, ҳамкасбим қўнғироқ қилиб, ҳаммасини тушунтиради."
    ),
    "ru": (
        "У меня нет точной информации по этому вопросу. Оставьте, пожалуйста, "
        "номер телефона — коллега перезвонит и всё расскажет."
    ),
}

# The clinic's own language, kept under a name because callers and tests
# refer to "the" no-match reply.
NO_MATCH_RESPONSE = NO_MATCH_RESPONSES["uz-latn"]


def no_match_response(user_message: str) -> str:
    """The no-match line, in the script the patient just wrote in."""
    return NO_MATCH_RESPONSES[reply_script(user_message)]


# Telegram's own buttons, not something a patient typed. "/start" is what the
# client sends when somebody opens the chat and presses Start, and it is the
# very first thing the clinic ever receives from most patients. "/help" is the
# other button the client offers from its menu, and it gets its own answer
# below rather than the greeting: a patient who presses it halfway through a
# conversation has asked what this chat can do, not arrived for the first time.
_CLIENT_COMMANDS = frozenset({"/start", "/help"})

# What Telegram puts after "/start" when a patient arrives through a deep link
# (t.me/<bot>?start=<payload>) -- the standard way an ad campaign says which
# ad the patient came from. Telegram restricts the payload to this alphabet
# and 64 characters, so the shape is worth matching on: the alternative is
# handing "clinic_ad_2" to the model and asking it to answer an opaque token.
#
# Deliberately narrow: only "/start" (never "/help") and only when the payload
# is the single thing that follows it. "/start bugun qabulga yozilsam" is a
# patient talking and still goes to the model. The one case this reads as a
# campaign tag when it wasn't is a patient who typed exactly one plain word
# after the command, and a greeting is a fair answer to that anyway.
_DEEP_LINK_PAYLOAD = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Answered from here rather than by the model, for three reasons.
#
# It is not a question, so there is nothing to answer: asked to reply to
# "/start", the model produces a greeting, which is the right words by
# accident rather than because it understood anything.
#
# It has no language in it. Script detection sees no Cyrillic and falls to
# Uzbek Latin, but the model is free to answer in either alphabet -- and did,
# greeting patients in Cyrillic and then, a moment later, again in Latin when
# their real "Salom" arrived. Two hellos in two alphabets is the clinic's
# first impression.
#
# And it costs a model call, at the one moment in a conversation where the
# reply is entirely predictable. On a deployment with a daily allowance, that
# call is better spent on the patient's actual question.
START_RESPONSES = {
    "uz-latn": (
        "Assalom alaykum! Sizni nima bezovta qilyapti — yoki shifokor qabuliga yozilmoqchimisiz?"
    ),
    "uz-cyrl": (
        "Ассалом алайкум! Сизни нима безовта қиляпти — ёки шифокор қабулига ёзилмоқчимисиз?"
    ),
    "ru": ("Здравствуйте! Что вас беспокоит — или хотите записаться на приём к врачу?"),
}

# "/help" asks what this chat can do, which is a different question from
# "hello". Answered from here for the same three reasons as the greeting --
# it is a button rather than a sentence, it carries no language, and its
# answer is fixed -- but it must not be the greeting itself: a patient who
# presses Help mid-conversation would otherwise be welcomed as if they had
# just arrived, and whatever they wanted help with is dropped.
HELP_RESPONSES = {
    "uz-latn": (
        "Men klinikaning yordamchisiman. Xizmatlar, shifokorlar va ish vaqti "
        "haqida so'rashingiz yoki shifokor qabuliga yozilishingiz mumkin. "
        "Sizni nima qiziqtiryapti?"
    ),
    "uz-cyrl": (
        "Мен клиниканинг ёрдамчисиман. Хизматлар, шифокорлар ва иш вақти "
        "ҳақида сўрашингиз ёки шифокор қабулига ёзилишингиз мумкин. "
        "Сизни нима қизиқтиряпти?"
    ),
    "ru": (
        "Я помощник клиники. Можете спросить об услугах, врачах и часах работы "
        "или записаться на приём. Что вас интересует?"
    ),
}

# How DEFAULT_REPLY_LANGUAGE (free text, named in English -- see
# app.core.config) picks one of the three scripts the fixed lines exist in.
# A language with no entry here, English included, falls back to the script
# the patient's own message was written in: there is no English greeting to
# fall back to, and the clinic's own language beats guessing.
_DEFAULT_LANGUAGE_SCRIPTS = {
    "russian": "ru",
    "ru": "ru",
    "русский": "ru",
    "uzbek": "uz-latn",
    "uz": "uz-latn",
    "o'zbek": "uz-latn",
    "oʻzbek": "uz-latn",
}


def _fixed_line_script(user_message: str, default_language: str | None) -> str:
    """Which script to answer a chat client's button in.

    A button carries no language -- "/start" has no letters at all, and a deep
    link's payload is a campaign tag, not something the patient wrote -- so
    reply_script() would answer every one of them in Uzbek Latin regardless of
    how the clinic is configured. The clinic's own DEFAULT_REPLY_LANGUAGE is
    the better signal here, and the message itself is only consulted when that
    setting names a language these fixed lines don't exist in.
    """
    if default_language is not None:
        script = _DEFAULT_LANGUAGE_SCRIPTS.get(default_language.strip().lower())
        if script is not None:
            return script
    return reply_script(user_message)


def client_command(user_message: str) -> str | None:
    """Which button the chat client sent on the patient's behalf, if any.

    In groups Telegram addresses commands to a particular bot ("/start@name"),
    and a client may pad them, so the text is trimmed to the bare command
    before it is compared. A patient who writes past the command has said
    something and is left to the model -- "/start bugun qabulga yozilsam" is a
    request, and swallowing it would lose the only thing they wrote.

    The one exception is a deep link's campaign tag; see _DEEP_LINK_PAYLOAD.
    """
    stripped = user_message.strip()
    if "\n" in stripped:
        return None

    parts = stripped.split()
    if not parts:
        return None

    command = parts[0].split("@", 1)[0].lower()
    if command not in _CLIENT_COMMANDS:
        return None
    if len(parts) == 1:
        return command
    if command == "/start" and len(parts) == 2 and _DEEP_LINK_PAYLOAD.match(parts[1]):
        return command
    return None


def is_client_command(user_message: str) -> bool:
    """Whether this "message" is a button the chat client sent on the
    patient's behalf.
    """
    return client_command(user_message) is not None


def client_command_response(user_message: str, default_language: str | None = None) -> str:
    """The fixed answer to a chat client's button, in the clinic's language.

    Raises KeyError if the message is not a client command -- callers reach
    this only through client_command(), which has already said that it is.
    """
    command = client_command(user_message)
    responses = HELP_RESPONSES if command == "/help" else START_RESPONSES
    return responses[_fixed_line_script(user_message, default_language)]


def start_response(user_message: str, default_language: str | None = None) -> str:
    """The opening line, in the clinic's own language."""
    return START_RESPONSES[_fixed_line_script(user_message, default_language)]


def _format_faq_context(matches: Sequence[KnowledgeBaseMatch]) -> str:
    if not matches:
        return "(No matching FAQ entries were found for this question.)"
    return "\n\n".join(
        f"Q: {match.knowledge_base.question}\nA: {match.knowledge_base.answer}" for match in matches
    )


def _doctor_roster(doctors: Sequence[Doctor]) -> str:
    """The clinic's clinicians, as prompt lines -- or "" when none are listed.

    "Kim qabul qiladi?" is one of the first things a patient asks, and until
    now the only honest answer was that we did not know: the roster was read
    on every message to work out how many bookings a slot holds, and then
    thrown away. Both rule 1s forbid stating the clinic's staff, so the model
    was refusing a question the database could answer.

    Rendered from the live rows rather than written into a FAQ entry, because
    a roster is not a fact that stays still. A doctor who leaves is
    deactivated in the dashboard, and this stops naming them on the next
    message -- where a seeded answer would keep offering a clinician who no
    longer works there until somebody remembered to edit it.

    Only what the front desk would say out loud: who they are, what they do,
    when they are in. Not the phone number on the row, which is the clinic's
    internal line to the doctor.
    """
    if not doctors:
        return ""
    # Six clinicians who all work the same day produced six identical
    # "09:00 - 18:00" tails, and a model copying the block back wrote the
    # hours six times in one reply. When the roster agrees with itself the
    # hours are a fact about the clinic, so they are stated as one.
    shared_hours = {doctor.working_hours for doctor in doctors}
    if len(shared_hours) == 1:
        lines = "\n".join(f"- {doctor.name} — {doctor.specialty}" for doctor in doctors)
        hours_note = (
            f"All of them are in during the clinic's usual hours ({shared_hours.pop()}), "
            "so those hours are a fact about the clinic and not about any one "
            "doctor: say them once if they are asked for, and never after each "
            "name.\n"
        )
    else:
        lines = "\n".join(
            f"- {doctor.name} — {doctor.specialty} — {doctor.working_hours}" for doctor in doctors
        )
        hours_note = ""
    return (
        "\n\nThe clinicians currently seeing patients here, given to you as "
        "fact:\n"
        f"{lines}\n"
        f"{hours_note}"
        "You may name them and say what each one does and when they work. "
        "This is the whole list: never name a doctor who is not on it, and "
        "never add anything to what is written about the ones who are — not "
        "their experience, their qualifications, where they studied, which "
        "conditions they are best with, nor any opinion of which of them a "
        "patient should see. You do not know those things, and a patient "
        "chooses a clinician on exactly that kind of claim.\n"
        "Working hours here mean the days and times that doctor is in the "
        "building. They are not free appointment times, and you have no way "
        "of knowing what is free: never read an hour out as though it were "
        "an available slot, and never promise a patient a particular doctor "
        "at a particular time. Who sees them, and when, is settled by the "
        "front desk on the telephone."
    )


def _clinic_facts_block(
    clinic_address: str | None,
    clinic_phone_numbers: str | None,
    doctors: Sequence[Doctor] = (),
    clinic_work_hours: str | None = None,
) -> str:
    """The handful of clinic facts that come from configuration and from the
    clinic's own tables rather than from the knowledge base, rendered as a
    prompt section -- or "" when there are none.

    Both prompts otherwise forbid stating an address, a phone number or who
    works here at all, which is the right default when the only source is
    retrieval: a made-up address sends a patient across Tashkent to a
    building that isn't there, and an invented doctor is worse still, because
    the patient arrives asking for somebody by name.

    These are exempt because an operator typed them, not because the model
    knows them, so they are presented as given facts and rule 1 on each path
    is written to permit exactly what appears here and nothing more.
    """
    lines: list[str] = []
    if clinic_address:
        lines.append(f"Address: {clinic_address}")
    if clinic_phone_numbers:
        lines.append(f"Phone: {clinic_phone_numbers}")
    if clinic_work_hours:
        lines.append(f"Open: {clinic_work_hours}")

    roster = _doctor_roster(doctors)
    if not lines:
        # The closing "these are the only details" sentence belongs to the
        # address-and-phone list and would be false standing on its own, so
        # with no configured details the roster is the whole block.
        return roster

    detail_lines = "\n".join(lines)
    return (
        "\n\nThese clinic details are given to you as fact. State them to a "
        "patient who asks, written out in their own language and alphabet, and "
        "never altered or added to:\n"
        f"{detail_lines}\n"
        "These are the only details of this clinic you have been given "
        "directly. Do not treat anything else about it as known on the "
        "strength of them." + roster
    )


def _price_contact_clause(clinic_phone_numbers: str | None) -> tuple[str, str, str]:
    """The three forms rules 3, 6 and 8 need of "ring the clinic": the Uzbek
    sentence, an English gloss so the model can say the same thing in Russian
    rather than pasting Uzbek at a Russian speaker, and the bare number for
    the rules that only need to drop it into a sentence of their own.

    This used to be a fallback for the prices the assistant could not find.
    It is now the answer to both prices and appointments, because the clinic
    quotes and books at the front desk and nowhere else.

    With no number configured there is nothing to send them to, so the offer
    narrows to a callback and the gloss says outright that no clinic number
    is known -- an instruction not to invent one is worth more here than
    anywhere else in the prompt, since a plausible-looking +998 number is
    exactly what a model will happily produce.
    """
    if clinic_phone_numbers:
        return (
            f"ushbu telefon raqamiga qo'ng'iroq qiling: {clinic_phone_numbers}",
            f"tell them to ring the clinic's front desk on {clinic_phone_numbers}",
            clinic_phone_numbers,
        )
    callback = (
        "telefon raqamingizni va qachon gaplashish siz uchun qulay bo'lgan "
        "vaqtni qoldiring, o'sha vaqtda o'zimiz qo'ng'iroq qilamiz"
    )
    return (
        callback,
        "ask them to leave their number together with a time that suits them, and "
        "promise the clinic will call then. You have NOT been given a phone number "
        "for this clinic, so do not read one out and never invent one",
        "the clinic's front desk",
    )


def _build_system_prompt(
    matches: Sequence[KnowledgeBaseMatch],
    flagged_as_medical_advice: bool,
    default_language: str,
    clinic_phone_numbers: str | None,
    clinic_address: str | None,
    signals: ConversationSignals,
    doctors: Sequence[Doctor] = (),
    clinic_work_hours: str | None = None,
    unpriceable: bool = False,
) -> str:
    price_contact, price_contact_gloss, price_contact_bare = _price_contact_clause(
        clinic_phone_numbers
    )
    shared = {
        "default_language": default_language,
        "price_contact": price_contact,
        "price_contact_gloss": price_contact_gloss,
        "price_contact_bare": price_contact_bare,
        "clinic_facts": _clinic_facts_block(
            clinic_address, clinic_phone_numbers, doctors, clinic_work_hours
        ),
    }
    if matches:
        prompt = _SYSTEM_PROMPT_TEMPLATE.format(faq_context=_format_faq_context(matches), **shared)
    else:
        prompt = _NO_FAQ_SYSTEM_PROMPT.format(**shared)
    # Appended after the rules rather than before them: rules 6 and 7 refer
    # to this section by name, and a reader (or a model) meeting the facts
    # first has nothing to do with them yet.
    prompt += render_signals(signals)
    if flagged_as_medical_advice:
        prompt += _MEDICAL_ADVICE_REMINDER
    if unpriceable:
        prompt += _UNNAMED_SERVICE_REMINDER
    return prompt


async def generate_answer(
    session: AsyncSession,
    user_message: str,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
    guardrail_classifier: GuardrailClassifier | None = None,
    settings: Settings | None = None,
    history: Sequence[ChatMessage] | None = None,
) -> str:
    """Turn an incoming patient message into a reply: guardrail check, then
    (unless it's an emergency) retrieve relevant FAQs and ask the LLM to
    answer from that context.

    When retrieval finds nothing, the reply depends on ANSWER_WITHOUT_FAQ.
    Off (the default), the LLM is never asked at all and NO_MATCH_RESPONSE
    is returned, so it cannot invent a clinic detail. On, it answers from
    general knowledge under _NO_FAQ_SYSTEM_PROMPT, which still forbids
    clinic specifics and medical advice — for a deployment whose knowledge
    base isn't populated yet, where one fixed refusal to every message is
    worse than a general reply.

    `history` is the conversation's earlier turns, oldest first, and is what
    lets a patient say "va narxi qancha?" and be understood. Retrieval and
    the guardrail still run against `user_message` alone, since those judge
    what was just asked rather than the whole conversation. Callers get it
    from app.services.conversation.context_for_reply, which already excludes
    the messages being answered right now, so appending user_message here
    cannot repeat them.
    """
    resolved_settings = settings or get_settings()

    # Before the guardrail, because a chat client's button is not a sentence
    # for it to judge and cannot be an emergency.
    if is_client_command(user_message):
        return client_command_response(user_message, resolved_settings.default_reply_language)

    guardrail = evaluate_guardrail(user_message, guardrail_classifier)
    if guardrail.fixed_response is not None:
        return guardrail.fixed_response

    # A price question with no service in it is not retrievable (see
    # app.services.question_shape): the knowledge base is one template
    # repeated over every service, so the query matches the template and the
    # rows that come back are arbitrary. Skipping retrieval entirely is
    # cheaper than filtering afterwards and, more to the point, it is the
    # only way the model cannot read them out.
    unpriceable = names_nothing_to_price(user_message)
    matches = (
        []
        if unpriceable
        else await retrieve_relevant_faqs(
            session, user_message, embedding_provider=embedding_provider
        )
    )
    if not matches and not unpriceable and not resolved_settings.answer_without_faq:
        # Code-level guarantee, not just a prompt instruction: if retrieval
        # found nothing (no rows, or every candidate fell beyond
        # retrieve_relevant_faqs's distance threshold), we don't ask the LLM
        # to improvise. This holds even if the model ever fails to follow
        # rule 1 below. Rule 1 stays in the prompt anyway, for the case this
        # check *doesn't* cover: matches is non-empty but none of the
        # retrieved FAQs actually answer the specific thing the patient
        # asked — retrieval found something in the neighborhood, just not
        # the right thing.
        return no_match_response(user_message)

    # The roster, still: a patient who asks who works here gets an answer.
    # The appointment book is no longer read at all -- the clinic books by
    # telephone, so showing the assistant a diary it may not offer from would
    # be putting the temptation in front of it and trusting a rule to hold.
    doctors = await DoctorRepository(session).list_active()

    system_prompt = _build_system_prompt(
        matches,
        doctors=doctors,
        signals=read_signals(history, user_message),
        flagged_as_medical_advice=guardrail.category is GuardrailCategory.MEDICAL_ADVICE,
        default_language=resolved_settings.default_reply_language,
        clinic_phone_numbers=resolved_settings.clinic_phone_numbers,
        clinic_address=resolved_settings.clinic_address,
        clinic_work_hours=resolved_settings.clinic_work_hours,
        unpriceable=unpriceable,
    )
    provider = llm_provider or get_llm_provider()
    conversation: list[ChatMessage] = [
        *(history or []),
        ChatMessage(role="user", content=user_message),
    ]
    # Read once more on the way out. Everything above this line guards what
    # the model is asked; this guards what it said, which nothing did before.
    return review_reply(await provider.generate(system_prompt, conversation), user_message)
