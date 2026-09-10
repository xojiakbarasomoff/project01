from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types as genai_types

from app.core.config import Settings
from app.rag.embeddings import GeminiEmbeddingProvider, _select_embedding_provider
from app.rag.llm import (
    HF_ROUTER_BASE_URL,
    GeminiLLMProvider,
    OpenAILLMProvider,
    QwenLLMProvider,
    _select_llm_provider,
)

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    gemini_api_key="test-gemini-key",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeCompletionsResource:
    def __init__(self, content: str | None) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        )


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletionsResource(content)


class _FakeAsyncOpenAI:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChat(content)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point llm.py imports it, so OpenAILLMProvider.__init__ picks up the
# fake client instead of a real network client.


async def test_generate_prepends_system_prompt_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAsyncOpenAI("Sure, we're open 9 to 5!")
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    result = await provider.generate(
        "You are a helpful assistant.", [{"role": "user", "content": "What are your hours?"}]
    )

    assert result == "Sure, we're open 9 to 5!"
    fake_client.chat.completions.create.assert_awaited_once_with(
        model=TEST_SETTINGS.openai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are your hours?"},
        ],
    )


async def test_openai_model_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_MODEL is a dashboard lever, like GEMINI_MODEL: a model that
    retires or answers badly has to be swappable while patients are waiting,
    not after a redeploy. It was pinned in code to gpt-4o-mini until now.
    """
    fake_client = _FakeAsyncOpenAI("ok")
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    settings = TEST_SETTINGS.model_copy(update={"openai_model": "gpt-5.6-luna"})
    provider = OpenAILLMProvider(settings=settings)
    await provider.generate("system", [{"role": "user", "content": "hi"}])

    assert fake_client.chat.completions.create.await_args.kwargs["model"] == "gpt-5.6-luna"


async def test_generate_raises_when_content_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI(None)
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    with pytest.raises(ValueError, match="no text content"):
        await provider.generate("system", [{"role": "user", "content": "hi"}])


# --- Gemini ---
# No test here ever talks to the real Gemini API: genai.Client is
# monkeypatched at the point llm.py imports it, so GeminiLLMProvider.__init__
# picks up the fake client instead of a real network client.


class _FakeGeminiModels:
    def __init__(self, text: str | None) -> None:
        self.generate_content = AsyncMock(return_value=SimpleNamespace(text=text))


class _FakeGeminiAio:
    def __init__(self, text: str | None) -> None:
        self.models = _FakeGeminiModels(text)


class _FakeGeminiClient:
    def __init__(self, text: str | None) -> None:
        self.aio = _FakeGeminiAio(text)


async def test_gemini_generate_maps_roles_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeGeminiClient("Sure, we're open 9 to 5!")
    monkeypatch.setattr("app.rag.llm.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiLLMProvider(settings=TEST_SETTINGS)
    result = await provider.generate(
        "You are a helpful assistant.",
        [
            {"role": "user", "content": "What are your hours?"},
            {"role": "assistant", "content": "Let me check."},
        ],
    )

    assert result == "Sure, we're open 9 to 5!"
    fake_client.aio.models.generate_content.assert_awaited_once_with(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Content(role="user", parts=[genai_types.Part(text="What are your hours?")]),
            # "assistant" maps to Gemini's "model" role, not "assistant".
            genai_types.Content(role="model", parts=[genai_types.Part(text="Let me check.")]),
        ],
        config=genai_types.GenerateContentConfig(system_instruction="You are a helpful assistant."),
    )


async def test_gemini_generate_raises_when_text_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeGeminiClient(None)
    monkeypatch.setattr("app.rag.llm.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiLLMProvider(settings=TEST_SETTINGS)
    with pytest.raises(ValueError, match="no text content"):
        await provider.generate("system", [{"role": "user", "content": "hi"}])


# --- provider selection ---


def test_select_llm_provider_returns_openai_when_configured() -> None:
    settings = Settings(
        database_url=TEST_SETTINGS.database_url,
        redis_url=TEST_SETTINGS.redis_url,
        webhook_verify_token=TEST_SETTINGS.webhook_verify_token,
        meta_app_secret=TEST_SETTINGS.meta_app_secret,
        model_provider="openai",
        openai_api_key="sk-test",
    )
    assert isinstance(_select_llm_provider(settings), OpenAILLMProvider)


def test_select_llm_provider_returns_gemini_when_configured() -> None:
    settings = Settings(
        database_url=TEST_SETTINGS.database_url,
        redis_url=TEST_SETTINGS.redis_url,
        webhook_verify_token=TEST_SETTINGS.webhook_verify_token,
        meta_app_secret=TEST_SETTINGS.meta_app_secret,
        model_provider="gemini",
        gemini_api_key="test-gemini-key",
    )
    assert isinstance(_select_llm_provider(settings), GeminiLLMProvider)


def test_select_llm_provider_defaults_to_gemini() -> None:
    assert isinstance(_select_llm_provider(TEST_SETTINGS), GeminiLLMProvider)


# --- constructing a provider without its key raises (defense in depth,
# independent of Settings' own model_provider/key validator) ---


def test_openai_llm_provider_without_key_raises() -> None:
    settings = TEST_SETTINGS.model_copy(update={"openai_api_key": None})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLMProvider(settings=settings)


def test_gemini_llm_provider_without_key_raises() -> None:
    settings = TEST_SETTINGS.model_copy(update={"gemini_api_key": None})
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLMProvider(settings=settings)


async def test_gemini_asks_the_model_named_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment whose daily allowance for one model is spent has to be
    able to move to another without shipping code, so the model name has to
    come from configuration rather than from the default argument.
    """
    fake_client = _FakeGeminiClient("Ha, albatta!")
    monkeypatch.setattr("app.rag.llm.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiLLMProvider(
        settings=TEST_SETTINGS.model_copy(update={"gemini_model": "gemini-3.1-flash-lite"})
    )
    await provider.generate("system", [{"role": "user", "content": "salom"}])

    assert fake_client.aio.models.generate_content.await_args.kwargs["model"] == (
        "gemini-3.1-flash-lite"
    )


def test_gemini_model_defaults_to_the_pinned_one() -> None:
    """Unset, nothing moves: the default is still the concrete version that
    was verified against the real API, not an alias.
    """
    assert TEST_SETTINGS.gemini_model == "gemini-2.5-flash"


# --- Qwen, through Hugging Face's OpenAI-compatible router ---


def test_qwen_talks_to_the_hugging_face_router_with_the_hugging_face_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is the OpenAI client, pointed elsewhere — so the thing worth
    asserting is that it is pointed elsewhere, and with the right credential.
    Sending an OpenAI key to Hugging Face, or an HF token to OpenAI, fails at
    the first patient message rather than here.
    """
    captured: dict[str, object] = {}

    def _fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", _fake_client)

    QwenLLMProvider(settings=TEST_SETTINGS.model_copy(update={"hf_token": "hf_test"}))

    assert captured["api_key"] == "hf_test"
    assert captured["base_url"] == HF_ROUTER_BASE_URL


def test_qwen_without_a_token_says_which_variable_is_missing() -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        QwenLLMProvider(settings=TEST_SETTINGS)


def test_llm_provider_overrides_the_backend_that_writes_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: object())
    settings = TEST_SETTINGS.model_copy(update={"llm_provider": "qwen", "hf_token": "hf_test"})

    assert isinstance(_select_llm_provider(settings), QwenLLMProvider)


def test_overriding_the_reply_backend_leaves_embeddings_where_they_are() -> None:
    """The point of the override. Moving embeddings invalidates every vector
    in knowledge_base, so a deployment switching the model that writes
    replies must not drag the clinic's FAQ along with it.
    """
    monkeypatch_free = TEST_SETTINGS.model_copy(
        update={"llm_provider": "qwen", "hf_token": "hf_test"}
    )

    assert monkeypatch_free.model_provider == "gemini"
    assert isinstance(_select_embedding_provider(monkeypatch_free), GeminiEmbeddingProvider)


def test_unset_llm_provider_follows_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing moves unless somebody sets it."""
    fake_client = _FakeGeminiClient("ok")
    monkeypatch.setattr("app.rag.llm.genai.Client", lambda **kwargs: fake_client)

    assert TEST_SETTINGS.llm_provider is None
    assert isinstance(_select_llm_provider(TEST_SETTINGS), GeminiLLMProvider)
