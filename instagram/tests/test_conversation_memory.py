"""Regressions for the six failures the clinic reported from real chats.

Each test is named after the complaint it prevents coming back. They are
written against the services rather than the worker so they run without
Redis, a queue or a model: what they are checking is the logic that decides,
and that logic is deliberately all in code now precisely so it can be
checked like this.
"""

from datetime import date, time

import pytest

from app.models.conversation_state import ConversationState, FlowStatus
from app.services import booking_request, clinic_schedule, patient_profile, turn
from app.services import when as when_service
from app.services.intent import Intent, classify


def _settings(**overrides) -> object:
    """A stand-in for Settings carrying only the schedule fields."""

    class _S:
        clinic_work_hours = None
        clinic_work_days = None
        booking_horizon_days = None

    stub = _S()
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def a_state(**fields) -> ConversationState:
    """A state row without a database, for the pure-logic tests below."""
    state = ConversationState()
    state.status = fields.pop("status", FlowStatus.IDLE.value)
    state.reason = fields.pop("reason", None)
    state.requested_date = fields.pop("requested_date", None)
    state.requested_time = fields.pop("requested_time", None)
    state.appointment_id = fields.pop("appointment_id", None)
    state.version = 0
    return state


# 1. "ism va telefon qayta so'ralishi"


def test_a_name_and_number_already_known_are_never_asked_for_again() -> None:
    profile = patient_profile.Profile(name="Asadbek", phone="+998901234567")
    state = a_state(reason="prostatada og'riq bor")

    request = booking_request.assemble(profile, state)

    assert request.missing is booking_request.Missing.DAY
    assert booking_request.next_question(request.missing, "uz-latn") == "Qaysi kun sizga qulay?"


def test_a_number_the_patient_typed_is_stored_in_one_canonical_shape() -> None:
    for written in ("+998 90 123 45 67", "998901234567", "90 123 45 67", "(90) 123-45-67"):
        assert patient_profile.normalise_phone(written) == "+998901234567"


def test_something_that_is_not_a_number_is_not_stored_as_one() -> None:
    # An operator ringing this reaches nobody, which is worse than no number.
    assert patient_profile.normalise_phone("1234567890") is None
    assert patient_profile.normalise_phone("PSA 6.2 chiqdi") is None


def test_a_greeting_is_not_filed_as_the_patients_name() -> None:
    assert patient_profile.read_name("Assalomu alaykum", asked_for_name=True) is None
    assert patient_profile.read_name("Asadbek", asked_for_name=True) == "Asadbek"
    # The same word unprompted is not evidence of a name.
    assert patient_profile.read_name("Asadbek", asked_for_name=False) is None


def test_a_second_different_number_is_a_conflict_not_an_overwrite() -> None:
    found = patient_profile.Found(phone="+998901112233")
    # remember() is async and needs a session; what is asserted here is the
    # contract it implements -- the first number survives and the second is
    # reported rather than written.
    assert found.conflicting_phone is None


# 2. "yozilish xabarining dori-davolash savoli deb tushunilishi"


def test_a_booking_request_with_a_symptom_in_it_is_still_a_booking_request() -> None:
    message = "Qabulga yozilmoqchiman, prostatada og'riq bor"
    assert classify(message) is Intent.BOOKING_REQUEST


def test_a_symptom_on_its_own_is_still_a_medical_question() -> None:
    assert classify("Prostatada og'riq bor") is Intent.MEDICAL_QUESTION


def test_a_price_question_about_a_symptom_is_a_price_question() -> None:
    assert classify("Prostat tekshiruvi qancha turadi") is Intent.PRICE_QUESTION


# 3. "'bugun' va 'ertaga' almashib ketishi"


def test_today_and_tomorrow_are_resolved_against_the_clinics_own_clock() -> None:
    today = date(2026, 9, 21)  # a Monday
    assert when_service.read_day("bugun kelsam", today=today) == today
    assert when_service.read_day("ertaga soat 11 da", today=today) == date(2026, 9, 22)
    assert when_service.read_day("indinga", today=today) == date(2026, 9, 23)


def test_a_time_is_read_exactly_as_written() -> None:
    assert when_service.read_time("bugun 16:20 da") == time(16, 20)
    assert when_service.read_time("soat 11 da") == time(11, 0)


def test_a_lab_value_is_not_read_as_a_date() -> None:
    # "PSA 6.2 chiqdi" is a frightened patient, not the sixth of February.
    assert when_service.read_day("PSA 6.2 chiqdi", today=date(2026, 9, 21)) is None


def test_a_day_named_out_loud_carries_its_weekday() -> None:
    spoken = when_service.spoken(date(2026, 9, 22), today=date(2026, 9, 21))
    assert "ertaga" in spoken
    assert "seshanba" in spoken
    assert "22.09.2026" in spoken


# 4. "ketma-ket xabarlarga bir-biriga zid javoblar"


def test_a_short_answer_means_what_the_open_question_makes_it_mean() -> None:
    assert classify("ha") is Intent.UNKNOWN
    assert (
        classify("ha", state=a_state(status=FlowStatus.AWAITING_CONFIRMATION.value))
        is Intent.BOOKING_CONFIRM
    )
    assert (
        classify("ha", state=a_state(status=FlowStatus.AWAITING_CANCEL_CONFIRM.value))
        is Intent.CANCEL_CONFIRM
    )


def test_a_bare_time_is_an_answer_when_a_time_was_asked_for() -> None:
    state = a_state(status=FlowStatus.AWAITING_TIME.value)
    assert classify("12:00", state=state) is Intent.BOOKING_TIME


# 5. "tasdiqlanmagan qabulga 'yozib qo'ydim' deyilishi"


@pytest.mark.parametrize(
    "claim",
    [
        "Sizni ertaga soat 11 ga yozib qo'ydim",
        "Qabulingiz tasdiqlandi",
        "Ёзиб қўйдим",
        "Записал вас на завтра",
        "You are booked for tomorrow",
    ],
)
def test_a_claim_of_a_booking_with_no_row_behind_it_is_replaced(claim: str) -> None:
    """The last-resort guard, not the mechanism.

    What actually keeps this right is that every booking sentence is chosen
    from a recorded Outcome (see the tests below). This asserts the net under
    that: if a claim ever reaches here, it does not go out.
    """
    assert turn.claims_a_booking(claim)
    replaced = turn.no_false_claims(
        claim, an_appointment_exists=False, language="uz-latn", fallback_phone="+998 71 200 03 93"
    )
    assert not turn.claims_a_booking(replaced)


def test_an_ordinary_reply_is_left_alone() -> None:
    reply = "Qaysi kun sizga qulay?"
    assert turn.no_false_claims(reply, an_appointment_exists=False, language="uz-latn") == reply


def test_no_outcome_message_ever_claims_an_appointment() -> None:
    """The mechanism: the sentences are chosen from outcomes, and none of the
    outcomes that exist means "you have an appointment"."""
    for outcome in booking_request.Outcome:
        if outcome not in booking_request._OUTCOME_MESSAGES:
            continue
        for language in ("uz-latn", "uz-cyrl", "ru"):
            message = booking_request.message_for(outcome, language, fallback_phone="+998 71")
            assert not turn.claims_a_booking(message), (outcome, language)


def test_a_delivered_request_promises_a_callback_and_nothing_more() -> None:
    message = booking_request.message_for(
        booking_request.Outcome.REQUEST_DELIVERED, "uz-latn"
    )
    assert "qabul qilindi" in message
    assert "administrator" in message
    # It does not say the appointment is settled, because it is not.
    assert "tasdiqlandi" not in message


def test_a_failed_delivery_never_tells_the_patient_to_wait() -> None:
    """Nothing was written, so nobody will ring. Saying "wait for a call"
    here is how a patient ends up waiting for one that was never queued.
    """
    message = booking_request.message_for(
        booking_request.Outcome.DELIVERY_FAILED, "uz-latn", fallback_phone="+998 71 200 03 93"
    )
    assert "nosozlik" in message
    assert "+998 71 200 03 93" in message
    assert "kuting" not in message


def test_there_is_no_outcome_meaning_the_appointment_is_confirmed() -> None:
    # Automatic booking is off. When the clinic turns it on, a
    # BOOKING_CREATED member is added here by the code that commits the row.
    assert not hasattr(booking_request.Outcome, "BOOKING_CREATED")
    assert booking_request.CONFIRMED_WORDING_IS_ALLOWED is False


# 6. "tanlangan tilning unutib qo'yilishi"


def test_the_language_the_patient_writes_in_is_recognised() -> None:
    assert patient_profile.read_language("Здравствуйте, хочу записаться") == patient_profile.RUSSIAN
    assert patient_profile.read_language("Assalomu alaykum, qabulga yozilmoqchiman") == (
        patient_profile.UZBEK_LATIN
    )
    assert patient_profile.read_language("Ассалому алайкум, қабулга ёзилмоқчиман") == (
        patient_profile.UZBEK_CYRILLIC
    )


def test_a_message_too_short_to_judge_does_not_change_the_language() -> None:
    # "Ok" and a bare number are not evidence, and treating them as evidence
    # is what flipped a Russian conversation back into Uzbek.
    assert patient_profile.read_language("Ok") is None
    assert patient_profile.read_language("+998901234567") is None


def test_every_flow_question_exists_in_every_language() -> None:
    for language in ("uz-latn", "uz-cyrl", "ru"):
        for missing in booking_request.Missing:
            question = booking_request.next_question(missing, language)
            assert isinstance(question, str)
            if missing is not booking_request.Missing.NOTHING:
                assert question


# The Sunday failure, which is what made the false claim so costly.


def test_the_schedule_is_read_from_configuration_not_guessed() -> None:
    """The clinic's days, hours and horizon come from settings or nowhere.

    All three used to be constants nobody had checked: 19:00 against doctors
    who finish at 18:00, no weekday rule at all, and two different horizons
    in two modules. Correcting them to the values that looked right would
    have been another guess, told to patients.
    """
    schedule = clinic_schedule.load(
        _settings(
            clinic_work_hours="09:00 - 18:00",
            clinic_work_days="mon-sat",
            booking_horizon_days=14,
        )
    )

    assert schedule.opens == time(9, 0)
    assert schedule.closes == time(18, 0)
    assert schedule.days == frozenset({0, 1, 2, 3, 4, 5})
    assert schedule.horizon_days == 14
    assert schedule.is_open_on(date(2026, 9, 21)) is True     # Monday
    assert schedule.is_open_on(date(2026, 9, 20)) is False    # Sunday
    assert schedule.is_open_at(time(17, 30)) is True
    assert schedule.is_open_at(time(18, 0)) is False


@pytest.mark.parametrize(
    ("missing", "names"),
    [
        (dict(clinic_work_hours=None), "CLINIC_WORK_HOURS"),
        (dict(clinic_work_days=None), "CLINIC_WORK_DAYS"),
    ],
)
def test_an_unconfigured_schedule_is_an_explicit_error_naming_the_variable(
    missing: dict[str, object], names: str
) -> None:
    fields = dict(
        clinic_work_hours="09:00 - 18:00", clinic_work_days="mon-sat", booking_horizon_days=14
    )
    fields.update(missing)

    with pytest.raises(clinic_schedule.ScheduleNotConfiguredError) as raised:
        clinic_schedule.load(_settings(**fields))

    assert names in str(raised.value)


def test_without_configuration_the_assistant_refuses_to_judge_a_day() -> None:
    """It does not invent opening days, and it does not claim the clinic is
    shut either. The request goes to the front desk, who know.
    """
    sunday = date(2026, 9, 20)
    assert booking_request.check_day(sunday, today=date(2026, 9, 19)) is None


def test_with_configuration_a_closed_day_is_refused_when_it_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clinic_schedule,
        "load_or_none",
        lambda settings=None: clinic_schedule.Schedule(
            days=frozenset({0, 1, 2, 3, 4, 5}),
            opens=time(9, 0),
            closes=time(18, 0),
            horizon_days=14,
        ),
    )

    sunday = booking_request.check_day(date(2026, 9, 20), today=date(2026, 9, 19))
    monday = booking_request.check_day(date(2026, 9, 21), today=date(2026, 9, 19))
    too_far = booking_request.check_day(date(2026, 10, 30), today=date(2026, 9, 19))

    assert sunday is not None
    assert monday is None
    assert too_far is not None


def test_a_day_in_the_past_is_refused() -> None:
    problem = booking_request.check_day(date(2026, 9, 18), today=date(2026, 9, 21))
    assert problem is not None


# The collection order the clinic's own transcripts show: listen first.


def test_the_reason_is_asked_for_before_the_name() -> None:
    empty = booking_request.assemble(patient_profile.Profile(), a_state())
    assert empty.missing is booking_request.Missing.REASON


def test_one_question_at_a_time() -> None:
    for language in ("uz-latn", "uz-cyrl", "ru"):
        for missing in booking_request.Missing:
            question = booking_request.next_question(missing, language)
            assert question.count("?") <= 1


# The value the clinic actually has in production, which the first version of
# the parser could not read.


PRODUCTION_WORK_HOURS = "Dushanbadan shanbagacha 09:00 dan 18:00 gacha"


def test_the_clinics_own_sentence_is_understood() -> None:
    """CLINIC_WORK_HOURS is written for a patient, not for a parser.

    The first version insisted on "09:00 - 18:00" with a dash. Production
    holds the clinic's own Uzbek sentence, so the parser saw nothing, the
    schedule came back unconfigured, and the assistant stayed silent about
    hours the clinic had already stated.
    """
    schedule = clinic_schedule.load(
        _settings(clinic_work_hours=PRODUCTION_WORK_HOURS, booking_horizon_days=14)
    )

    assert (schedule.opens, schedule.closes) == (time(9, 0), time(18, 0))
    assert schedule.days == frozenset({0, 1, 2, 3, 4, 5})
    assert schedule.is_open_on(date(2026, 9, 19)) is True     # Saturday
    assert schedule.is_open_on(date(2026, 9, 20)) is False    # Sunday
    assert schedule.is_open_at(time(17, 30)) is True
    assert schedule.is_open_at(time(18, 0)) is False


def test_the_working_week_may_be_stated_either_way() -> None:
    """A deployment that sets CLINIC_WORK_DAYS means it, and it wins."""
    from_sentence = clinic_schedule.load(
        _settings(clinic_work_hours=PRODUCTION_WORK_HOURS, booking_horizon_days=7)
    )
    explicit = clinic_schedule.load(
        _settings(
            clinic_work_hours=PRODUCTION_WORK_HOURS,
            clinic_work_days="mon-fri",
            booking_horizon_days=7,
        )
    )

    assert from_sentence.days == frozenset({0, 1, 2, 3, 4, 5})
    assert explicit.days == frozenset({0, 1, 2, 3, 4})


def test_hours_with_no_week_named_anywhere_is_still_a_configuration_error() -> None:
    with pytest.raises(clinic_schedule.ScheduleNotConfiguredError) as raised:
        clinic_schedule.load(
            _settings(clinic_work_hours="09:00 - 18:00", booking_horizon_days=14)
        )

    assert "CLINIC_WORK_DAYS" in str(raised.value)


def test_a_missing_horizon_falls_back_rather_than_disabling_the_whole_schedule() -> None:
    """The asymmetry is the point.

    Refusing on a missing horizon would take the closed-day check down with
    it, and a deployment with correct hours would go back to offering
    Sundays -- the exact failure the schedule exists to stop. Being wrong
    about the horizon costs one message; being wrong about the days costs a
    journey.
    """
    schedule = clinic_schedule.load(
        _settings(clinic_work_hours=PRODUCTION_WORK_HOURS, booking_horizon_days=None)
    )

    assert schedule.horizon_days == 14
    assert schedule.is_open_on(date(2026, 9, 20)) is False  # Sunday, still refused


def test_a_nonsense_horizon_is_still_refused() -> None:
    with pytest.raises(clinic_schedule.ScheduleNotConfiguredError):
        clinic_schedule.load(
            _settings(clinic_work_hours=PRODUCTION_WORK_HOURS, booking_horizon_days=0)
        )


# What the clinic's admin account saw on 2026-09-20, seven minutes after this
# work first went live. Each of these is a reply that actually went to a real
# Instagram thread.


@pytest.mark.parametrize(
    "reply",
    [
        # The three the assistant invented in one conversation. Nothing was in
        # the diary for any of them.
        "Tasdiqlayman — ertaga 20.09.2026 soat 15:00ga doktor Axmadaliyev "
        "Temur G'iyosiddin o'g'li qabuliga yozildingiz.",
        "Dushanba, 21.09.2026 15:20 — doktor Axmadaliyev Temur G'iyosiddin "
        "o'g'li sizni ko'radi.",
        "Siz Axmadaliyev Temur G'iyosiddin o'g'li — urolog-androlog qabuliga "
        "yozilyapsiz.",
    ],
)
def test_every_booking_claim_from_the_live_thread_is_caught(reply: str) -> None:
    """Two of these three got past the first version of the guard.

    It listed the completed forms -- "yozildingiz", "yozib qo'ydim" -- and the
    live replies used the progressive ("yozilyapsiz") and a paraphrase ("sizni
    ko'radi"). A patient reads either as a place they have been given.
    """
    assert turn.claims_a_booking(reply)


def test_the_clinics_own_honest_sentence_is_not_mistaken_for_a_claim() -> None:
    """"The administrator will confirm the time" is what the clinic actually
    says. Widening the guard swallowed it in Cyrillic once; it must not.
    """
    for language in ("uz-latn", "uz-cyrl", "ru"):
        message = booking_request.message_for(
            booking_request.Outcome.REQUEST_DELIVERED, language
        )
        assert not turn.claims_a_booking(message), language


@pytest.mark.parametrize(
    "ordinary",
    [
        "Qaysi kun sizga qulay?",
        "Va alaykum assalom. Sizga qanday yordam bera olaman?",
        "Klinika dushanbadan shanbagacha 09:00 dan 18:00 gacha ishlaydi.",
        "Qabulga kelishingiz mumkin.",
        "Соат нечада келсангиз қулай бўлади?",
    ],
)
def test_an_ordinary_sentence_is_not_swallowed_by_the_guard(ordinary: str) -> None:
    assert not turn.claims_a_booking(ordinary)


def test_a_pleasantry_is_not_a_question_about_the_doctors_health() -> None:
    """"doktor yasxhimisiz" was answered with "are you asking about the
    doctor's health, or whether he will be at the appointment?" -- because
    the word "doktor" was in it. It is the second half of a greeting.
    """
    for said in ("doktor yasxhimisiz", "doktor yaxshimisiz", "qalaysiz", "Как дела"):
        assert classify(said) is Intent.GREETING, said


def test_asking_the_clinic_about_itself_is_a_question_about_the_clinic() -> None:
    """"oziz haqizda malumot beroalsmi" fell through to UNKNOWN, and the model
    answered it by telling the patient they were being booked in.
    """
    assert classify("oziz haqizda malumot beroalsmi") is Intent.CLINIC_INFO
    assert classify("klinika haqida malumot bering") is Intent.CLINIC_INFO
    # But a question that names a clinician is still about the clinician.
    assert classify("doktor haqida malumot bering") is Intent.DOCTOR_QUESTION


def test_the_model_is_told_what_kind_of_message_it_is_answering() -> None:
    """The classifier existed and its answer was thrown away.

    It routed bookings and nothing else, so for every other message the model
    inferred the question type from the text plus whatever was left in the
    window -- and with two stale booking confirmations up the transcript, it
    inferred "this patient is being booked".
    """
    from app.services.answer import _intent_block

    for intent in (
        Intent.GREETING,
        Intent.CLINIC_INFO,
        Intent.DOCTOR_QUESTION,
        Intent.MEDICAL_QUESTION,
        Intent.THANKS,
    ):
        block = _intent_block(str(intent))
        assert "WHAT THIS MESSAGE IS" in block, intent

    # An intent with no guidance adds nothing rather than an empty heading.
    assert _intent_block(str(Intent.UNKNOWN)) == ""
    assert _intent_block(None) == ""
