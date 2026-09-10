from abc import ABC, abstractmethod
from functools import lru_cache

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

# Dimension of whichever provider is actually configured (Settings.model_provider)
# — the knowledge_base.embedding column is a single fixed-width pgvector column,
# so only one provider's output size can be live at a time. Currently Gemini's
# gemini-embedding-001, truncated from its native 3072 via output_dimensionality
# (Matryoshka representation learning — a supported, intentional truncation, not
# a hack). 1536, not the native 3072: pgvector's HNSW/IVFFlat indexes hard-cap at
# 2000 dimensions (verified against pgvector 0.8.6), so 3072 can't be indexed at
# all; 1536 keeps a real ANN index with headroom to spare. Switching provider
# requires a migration to match, see migrations/versions.
EMBEDDING_DIMENSIONS = 1536


class EmbeddingProvider(ABC):
    """Abstraction over "turn text into vectors", mirroring the TZ's
    LLMProvider idea so the concrete backend (OpenAI, Gemini, something else
    later) can be swapped without touching callers. Callers should depend on
    this interface, not a concrete provider directly, so tests can inject a
    fake instead of hitting the network.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text, returning one vector per input in the same order."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        api_key = (settings or get_settings()).openai_api_key
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is required to use OpenAIEmbeddingProvider")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # A single batched call, not one call per text: OpenAI's embeddings
        # endpoint accepts a list under `input` and returns vectors in the
        # same order, so batching here saves N-1 round trips per ingest.
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return [item.embedding for item in response.data]


# Gemini rejects a batch embed of more than this many texts outright:
# "BatchEmbedContentsRequest.requests: at most 100 requests can be in one
# batch", a 400 rather than a truncation. Callers pass whatever they have --
# ingest_faqs hands over an entire FAQ file in one go -- so the limit is
# enforced here, where it is the API's rule rather than the caller's problem.
# A clinic with 98 FAQs never met it; one with 680 fails on the first deploy.
_GEMINI_BATCH_LIMIT = 100


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        model: str = "gemini-embedding-001",
    ) -> None:
        api_key = (settings or get_settings()).gemini_api_key
        if api_key is None:
            raise ValueError("GEMINI_API_KEY is required to use GeminiEmbeddingProvider")
        self._model = model
        self._client = genai.Client(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Batched the same way as OpenAIEmbeddingProvider: one call per
        # chunk, not one per text. gemini-embedding-001's native output is
        # 3072 dims — explicitly truncated to EMBEDDING_DIMENSIONS (1536) via
        # output_dimensionality, since pgvector can't index anything above
        # 2000 dims (see EMBEDDING_DIMENSIONS' comment).
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _GEMINI_BATCH_LIMIT):
            chunk = texts[start : start + _GEMINI_BATCH_LIMIT]
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=chunk,
                config=genai_types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            if response.embeddings is None:
                raise ValueError("Gemini embed_content returned no embeddings")
            vectors.extend(list(embedding.values or []) for embedding in response.embeddings)
        return vectors


def _select_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.model_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    return GeminiEmbeddingProvider(settings)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return _select_embedding_provider(get_settings())
