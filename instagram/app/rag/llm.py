from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, TypedDict, cast

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import Settings, get_settings


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMProvider(ABC):
    """Abstraction over "turn a system prompt + conversation into a reply",
    mirroring EmbeddingProvider so the backend/model can change without
    touching callers, and so tests can inject a fake instead of hitting the
    network.
    """

    @abstractmethod
    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        """Generate a reply given a system prompt and the conversation so far."""


class OpenAILLMProvider(LLMProvider):
    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        resolved = settings or get_settings()
        api_key = resolved.openai_api_key
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is required to use OpenAILLMProvider")
        # Settings.openai_model, not a literal default here, so the model can
        # be changed from the host's dashboard -- see GeminiLLMProvider, which
        # takes GEMINI_MODEL the same way and for the same reasons.
        self._model = model or resolved.openai_model
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        # ChatMessage is deliberately narrower than the SDK's message union
        # (only user/assistant; system is handled separately by this
        # abstraction), so it doesn't structurally unify with
        # ChatCompletionMessageParam under mypy strict. Both sides are
        # simple {role, content} dicts at runtime, so the cast is safe.
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": "system", "content": system_prompt}, *messages],
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=payload,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI chat completion returned no text content")
        return content


class GeminiLLMProvider(LLMProvider):
    # gemini-2.0-flash (originally proposed) is deprecated/no longer served.
    # gemini-2.5-flash verified live against the real API: responds
    # reliably. Pinned to a concrete version rather than an alias like
    # "gemini-flash-latest" (also verified working) — an alias can shift
    # which model actually runs without any change on our side, which is
    # bad for reproducing behavior/debugging later. The newer
    # gemini-3.7-flash was tried too and returned a 503 (overloaded) at
    # verification time — not reliable enough to default to.
    #
    # The default lives on Settings.gemini_model rather than here, so that a
    # deployment whose daily free-tier allowance for one model is spent can
    # be moved to another by setting GEMINI_MODEL, without waiting for a
    # code change to ship.
    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        resolved = settings or get_settings()
        api_key = resolved.gemini_api_key
        if api_key is None:
            raise ValueError("GEMINI_API_KEY is required to use GeminiLLMProvider")
        self._model = model or resolved.gemini_model
        self._client = genai.Client(api_key=api_key)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        # Gemini's turn roles are "user"/"model", not "user"/"assistant".
        contents = [
            genai_types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[genai_types.Part(text=message["content"])],
            )
            for message in messages
        ]
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(system_instruction=system_prompt),
        )
        if response.text is None:
            raise ValueError("Gemini generate_content returned no text content")
        return response.text


# Hugging Face's Inference Providers router, which speaks the OpenAI chat
# completions API. Reached through AsyncOpenAI with the base URL swapped
# rather than through a Hugging Face SDK: it is the same protocol, and one
# client with two base URLs is less to keep working than two clients.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


class QwenLLMProvider(LLMProvider):
    """Qwen, served through Hugging Face's OpenAI-compatible router.

    Kept as its own class rather than as an argument to OpenAILLMProvider
    because it reads a different credential and has a different default
    model, and because "which provider is this deployment on" should be
    answerable by looking at the type.
    """

    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        resolved = settings or get_settings()
        if resolved.hf_token is None:
            raise ValueError("HF_TOKEN is required to use QwenLLMProvider")
        self._model = model or resolved.qwen_model
        self._client = AsyncOpenAI(api_key=resolved.hf_token, base_url=HF_ROUTER_BASE_URL)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": "system", "content": system_prompt}, *messages],
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=payload,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Qwen chat completion returned no text content")
        return content


def _select_llm_provider(settings: Settings) -> LLMProvider:
    # LLM_PROVIDER, when set, overrides MODEL_PROVIDER for replies only —
    # embeddings stay where they are, because moving those means re-embedding
    # the knowledge base (see Settings.llm_provider).
    chosen = settings.llm_provider or settings.model_provider
    if chosen == "openai":
        return OpenAILLMProvider(settings)
    if chosen == "qwen":
        return QwenLLMProvider(settings)
    return GeminiLLMProvider(settings)


@lru_cache
def get_llm_provider() -> LLMProvider:
    return _select_llm_provider(get_settings())
