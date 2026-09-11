from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types as genai_types

from app.core.config import Settings
from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
    _select_embedding_provider,
)

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    gemini_api_key="test-gemini-key",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeEmbeddingsResource:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=vector) for vector in vectors]
            )
        )


class _FakeAsyncOpenAI:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _FakeEmbeddingsResource(vectors)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point embeddings.py imports it, so OpenAIEmbeddingProvider.__init__
# picks up the fake client instead of a real network client.


async def test_embed_returns_vectors_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
    fake_client = _FakeAsyncOpenAI(vectors)
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(["hello", "world"])

    assert result == vectors
    fake_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["hello", "world"],
        dimensions=EMBEDDING_DIMENSIONS,
    )


async def test_embed_empty_list_skips_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI([])
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed([])

    assert result == []
    fake_client.embeddings.create.assert_not_awaited()


# --- Gemini ---
# No test here ever talks to the real Gemini API: genai.Client is
# monkeypatched at the point embeddings.py imports it, so
# GeminiEmbeddingProvider.__init__ picks up the fake client instead of a
# real network client.


class _FakeGeminiEmbedModels:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embed_content = AsyncMock(
            return_value=SimpleNamespace(
                embeddings=[SimpleNamespace(values=vector) for vector in vectors]
            )
        )


class _FakeGeminiEmbedAio:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.models = _FakeGeminiEmbedModels(vectors)


class _FakeGeminiClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.aio = _FakeGeminiEmbedAio(vectors)


async def test_gemini_embed_returns_vectors_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
    fake_client = _FakeGeminiClient(vectors)
    monkeypatch.setattr("app.rag.embeddings.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(["hello", "world"])

    assert result == vectors
    fake_client.aio.models.embed_content.assert_awaited_once_with(
        model="gemini-embedding-001",
        contents=["hello", "world"],
        config=genai_types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )


class _EchoingGeminiModels:
    """Answers each call with one vector per text it was given, so a chunked
    embed can be checked for both call count and ordering.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed_content(self, *, model: str, contents: list[str], config: object):
        self.batches.append(list(contents))
        return SimpleNamespace(
            values=None,
            embeddings=[
                SimpleNamespace(values=[float(int(text))] * EMBEDDING_DIMENSIONS)
                for text in contents
            ],
        )


async def test_gemini_embed_splits_batches_over_the_api_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini rejects a batch of more than 100 outright ("at most 100 requests
    can be in one batch", a 400). Seeding a clinic's FAQ file is a single
    embed() of the whole file, so anything past 100 rows failed the deploy --
    which is what happened the first time a real price list was loaded.
    """
    models = _EchoingGeminiModels()
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=models))
    monkeypatch.setattr("app.rag.embeddings.genai.Client", lambda **kwargs: fake_client)

    texts = [str(index) for index in range(250)]
    provider = GeminiEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(texts)

    assert [len(batch) for batch in models.batches] == [100, 100, 50]
    assert len(result) == 250
    # Order survives the split: vector i still belongs to text i.
    assert [vector[0] for vector in result] == [float(index) for index in range(250)]


async def test_gemini_embed_empty_list_skips_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeGeminiClient([])
    monkeypatch.setattr("app.rag.embeddings.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed([])

    assert result == []
    fake_client.aio.models.embed_content.assert_not_awaited()


async def test_gemini_embed_raises_when_response_has_no_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeGeminiClient([[0.1] * EMBEDDING_DIMENSIONS])
    fake_client.aio.models.embed_content = AsyncMock(return_value=SimpleNamespace(embeddings=None))
    monkeypatch.setattr("app.rag.embeddings.genai.Client", lambda **kwargs: fake_client)

    provider = GeminiEmbeddingProvider(settings=TEST_SETTINGS)
    with pytest.raises(ValueError, match="no embeddings"):
        await provider.embed(["hello"])


# --- provider selection ---


def test_select_embedding_provider_returns_openai_when_configured() -> None:
    settings = Settings(
        database_url=TEST_SETTINGS.database_url,
        redis_url=TEST_SETTINGS.redis_url,
        webhook_verify_token=TEST_SETTINGS.webhook_verify_token,
        meta_app_secret=TEST_SETTINGS.meta_app_secret,
        model_provider="openai",
        openai_api_key="sk-test",
    )
    assert isinstance(_select_embedding_provider(settings), OpenAIEmbeddingProvider)


def test_select_embedding_provider_returns_gemini_when_configured() -> None:
    settings = Settings(
        database_url=TEST_SETTINGS.database_url,
        redis_url=TEST_SETTINGS.redis_url,
        webhook_verify_token=TEST_SETTINGS.webhook_verify_token,
        meta_app_secret=TEST_SETTINGS.meta_app_secret,
        model_provider="gemini",
        gemini_api_key="test-gemini-key",
    )
    assert isinstance(_select_embedding_provider(settings), GeminiEmbeddingProvider)


def test_select_embedding_provider_defaults_to_gemini() -> None:
    assert isinstance(_select_embedding_provider(TEST_SETTINGS), GeminiEmbeddingProvider)


# --- constructing a provider without its key raises (defense in depth,
# independent of Settings' own model_provider/key validator) ---


def test_openai_embedding_provider_without_key_raises() -> None:
    settings = TEST_SETTINGS.model_copy(update={"openai_api_key": None})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(settings=settings)


def test_gemini_embedding_provider_without_key_raises() -> None:
    settings = TEST_SETTINGS.model_copy(update={"gemini_api_key": None})
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider(settings=settings)


class _EchoingOpenAIEmbeddings:
    """One vector per input text, so a chunked embed can be checked for both
    call count and ordering.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def create(self, *, model: str, input: list[str], dimensions: int):
        self.batches.append(list(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(int(text))] * EMBEDDING_DIMENSIONS)
                for text in input
            ]
        )


async def test_openai_embed_splits_batches_over_the_api_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI takes at most 2048 inputs per request. Gemini's much lower cap
    was hit immediately and fixed; this one sat unnoticed because the clinic
    was under it -- and the knowledge base has been growing a wording at a
    time, from 98 rows to over 1500. ingest_faqs embeds a whole file in one
    call, so the deploy that crosses 2048 loses its entire seed.
    """
    embeddings = _EchoingOpenAIEmbeddings()
    fake_client = SimpleNamespace(embeddings=embeddings)
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    texts = [str(index) for index in range(5000)]
    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(texts)

    assert [len(batch) for batch in embeddings.batches] == [2048, 2048, 904]
    assert len(result) == 5000
    # Order survives the split: vector i still belongs to text i.
    assert [vector[0] for vector in result] == [float(index) for index in range(5000)]
