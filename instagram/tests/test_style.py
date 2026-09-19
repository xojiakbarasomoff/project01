"""The clinic's voice corpus: how it was selected, and that it is actually used.

Two separate claims are checked here, because the second one is the one that
would otherwise be taken on trust: the corpus exists, and the assistant
reaches for it. A data file nothing reads is not a feature.

The third claim is the one the clinic cares about most, and it is a negative:
these old chats are tone, never memory. Nothing about the patient being
answered may come out of somebody else's conversation.
"""

import json
import re

from app.services import style


def test_nothing_is_shown_to_the_model_until_a_person_approves_it() -> None:
    """The export has no author field, so nothing in it can be attributed.

    `role` and `content` are the only keys. Every reply is labelled
    `assistant` because every reply left the clinic's one account -- used by
    the doctor, by an administrator, by whoever pasted the saved replies and
    possibly by an earlier bot. A heuristic produces a shortlist, not a
    verdict, so the prompt gets the approved list and the approved list
    starts empty.
    """
    assert style.corpus() == ()
    assert style.choose("Qabulga yozilmoqchiman") == []
    assert style.render(style.choose("Salom")) == ""


def test_the_approved_list_is_the_only_thing_read_on_the_reply_path() -> None:
    assert style.CORPUS_PATH.name == "clinic_voice_approved.json"
    assert json.loads(style.CORPUS_PATH.read_text(encoding="utf-8")) == []


def test_there_is_a_shortlist_waiting_to_be_reviewed() -> None:
    candidates = json.loads(style.CANDIDATES_PATH.read_text(encoding="utf-8"))

    assert len(candidates) == 2015
    # Numbered, so a reviewer can say which ones they approved.
    assert [row["id"] for row in candidates[:3]] == [0, 1, 2]
    assert all(row["patient"] and row["clinic"] for row in candidates)


def test_no_patient_detail_survives_into_the_shortlist() -> None:
    """These sit in the repository and are read by a person. A telephone
    number left in one is a real patient's number.
    """
    candidates = json.loads(style.CANDIDATES_PATH.read_text(encoding="utf-8"))
    patterns = (
        ("number", r"\d[\d\s\-()]{6,}\d"),
        ("link", r"https?://"),
        ("handle", r"@[A-Za-z0-9._]{3,}|\b[A-Za-z][A-Za-z0-9.]*(?:_[A-Za-z0-9.]+)+\b"),
        ("name", r"(?:ismim|исмим|зовут)\s+[A-Za-zЀ-ӿ]{3,}"),
        ("age", r"\d{1,2}\s*(?:yosh|ёш|лет)"),
    )
    leaks = [
        (label, text[:80])
        for row in candidates
        for text in (row["patient"], row["clinic"])
        for label, pattern in patterns
        if re.search(pattern, text, re.I)
    ]

    assert leaks == []


def test_approved_examples_would_reach_the_prompt(monkeypatch, tmp_path) -> None:
    """The wiring works; only the input is withheld. Without this the empty
    list above would be indistinguishable from a feature that does nothing.
    """
    approved = tmp_path / "approved.json"
    approved.write_text(
        json.dumps([{"patient": "Qabulga yozilmoqchiman", "clinic": "Qaysi kun qulay?"}]),
        encoding="utf-8",
    )
    style.corpus.cache_clear()
    monkeypatch.setattr(style, "CORPUS_PATH", approved)
    try:
        chosen = style.choose("Qabulga yozilmoqchiman")
        rendered = style.render(chosen)

        assert len(chosen) == 1
        assert "HOW THIS CLINIC WRITES" in rendered
        assert "Copy the tone" in rendered
        assert "Do NOT copy any fact" in rendered
    finally:
        style.corpus.cache_clear()


def test_an_unreadable_corpus_does_not_stop_the_assistant_answering(monkeypatch) -> None:
    """Losing the tone is worth less than refusing to reply."""
    from pathlib import Path

    style.corpus.cache_clear()
    monkeypatch.setattr(style, "CORPUS_PATH", Path("does-not-exist.json"))
    try:
        assert style.corpus() == ()
        assert style.choose("Salom") == []
        assert style.render([]) == ""
    finally:
        style.corpus.cache_clear()


# --- and that it is not memory ----------------------------------------------


def test_the_corpus_is_not_consulted_for_anything_about_this_patient() -> None:
    """The name, number, language and request state come from the database.

    app.services.style has no session, no user id and no way to reach either
    -- which is the structural version of this promise, stronger than a
    comment saying so.
    """
    import inspect

    source = inspect.getsource(style)

    assert "AsyncSession" not in source
    assert "user_id" not in source
    assert "repositories" not in source


def test_turn_reads_the_patients_own_record_not_an_old_chat() -> None:
    import inspect

    from app.services import turn

    source = inspect.getsource(turn)

    assert "UserRepository" in source
    assert "ConversationStateRepository" in source
    assert "style" not in source
