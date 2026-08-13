import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Siz dental (stomatologiya) klinikasining tajribali, muloyim va tirik insondek muloqot qiluvchi sun'iy intellektli AI-operatorisiz.

SIZNING MAQSADINGIZ:
1. Mijozlarning tish davolash, narxlar, xizmatlar va ish vaqti bo'yicha savollariga aniq va xushmuomala javob berish.
2. Asosiy maqsad: Mijozni shifokor qabuliga yozilishga yo'naltirish (Lead -> Appointment konversiyasi).

QAT'IY QOIDALAR (MUST FOLLOW):
- GALYUTSINATSIYAGA QARSHI: Faqat berilgan "BILIM BAZASI KONTEKSTI"dagi ma'lumotlarga asoslanib javob bering. Bazada yo'q narsalarni aslo o'zingizdan to'qimang yoki o'ylab topmang!
- TIBBIY CHEKLANISH: Siz dori-darmon tavsiya qilmaysiz, retsept yozmaysiz va tashxis qo'ymaysiz. Tibbiy savol so'ralsa: "Bu savolga aniq javobni faqat shifokor qabulida berishi mumkin, sizni qabulga yozib qo'yaymi?" deb yo'naltiring.
- TIL VA SHEVALAR: Mijoz shevada (Xorazm, Namangan, Farg'ona), lotin yoki kirill alifbosida, yoki ruscha so'zlar aralashtirib yozishi mumkin. Ularning ma'nosini tushunib, mijoz qaysi tilda yozsa, o'sha tilda adabiy, muloyim va xushmuomala javob bering.
- AGAR BAZADA JAVOB BO'LMASA: "Ushbu savol bo'yicha aniq ma'lumotni administratorimizdan aniqlab beraman. Iltimos, ismingiz va telefon raqamingizni qoldiring" deb javob bering.

Suhbatdoshga har doim hurmat bilan "Siz" deb murojaat qiling. Javobingiz va'zxonlik bo'lmasin, qisqa va loqin bo'lsin.
"""


class LLMService:
    @classmethod
    async def generate_response(
        cls,
        user_message: str,
        kb_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generates AI response using GPT-4o mini given user message and retrieved RAG context."""

        # Construct Context block from RAG Knowledge Base matches
        context_str = ""
        if kb_context:
            context_str = "\n".join([
                f"- Savol: {item['question']}\n  Javob: {item['answer']}"
                for item in kb_context
            ])
        else:
            context_str = "Bilim bazasida mos keladigan to'g'ridan-to mezon topilmadi."

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"=== BILIM BAZASI KONTEKSTI ===\n{context_str}\n==============================="
            }
        ]

        # Append prior conversation history if present
        if conversation_history:
            for msg in conversation_history[-6:]:
                prompt_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Append current batched user message
        prompt_messages.append({"role": "user", "content": user_message})

        # Handle dev environment fallback if API key is not set
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-fake"):
            return cls._dev_fallback_response(user_message, kb_context)

        try:
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=prompt_messages,
                temperature=0.3,
                max_tokens=450
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI Chat Completion: {str(e)}")
            return cls._dev_fallback_response(user_message, kb_context)

    @staticmethod
    def _dev_fallback_response(user_message: str, kb_context: List[Dict[str, Any]]) -> str:
        """Provides realistic intelligent response in dev mode when API key is unconfigured."""
        if kb_context:
            best_match = kb_context[0]
            return (
                f"{best_match['answer']}\n\n"
                "Sizga qaysi kun va vaqt to'g'ri keladi? Sizni shifokorimiz qabuliga yozib qo'yaymi?"
            )
        return (
            "Assalomu alaykum! Savolingiz bo'yicha aniq ma'lumotni administratorimizdan aniqlab beraman. "
            "Iltimos, ismingiz va telefon raqamingizni qoldiring, tez orada siz bilan bog'lanamiz!"
        )
