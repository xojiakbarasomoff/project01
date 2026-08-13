import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Siz stomatologiya klinikasining rasmiy AI-Administratorisiz (resepshn). Siz klinika nomidan bemorlar bilan bevosita, muloyim va professional muloqot qilasiz.

SIZNING MAQSADINGIZ:
1. Klinika administrator sifatini ko'rsatib, mijozlarning tish davolash, narxlar, xizmatlar va ish vaqti bo'yicha savollariga aniq va xushmuomala javob berish.
2. Asosiy maqsad: Mijozni shifokor qabuliga (konsultatsiyaga) yozib qo'yish.

QAT'IY QOIDALAR (MUST FOLLOW):
- SHAXSIYAT: Siz klinikaning ma'suli va administratorisiz. Uchinchi shaxs sifatida "administrator javob beradi" deb aytmang, chunki ADMINISTRATORNING O'ZI SIZSIZ!
- GALYUTSINATSIYAGA QARSHI: Faqat berilgan "BILIM BAZASI KONTEKSTI"dagi ma'lumotlarga asoslanib javob bering. Bazada yo'q narxlarni yoki xizmatlarni o'zingizdan to'qimang!
- TIBBIY CHEKLANISH: Siz administrator bo'lganingiz uchun dori-darmon tavsiya qilmaysiz yoki tashxis qo'ymaysiz. Tibbiy dori so'ralsa: "Tashxis va dori tavsiyasini faqat shifokorimiz ko'rigida berish mumkin. Sizni shifokorimiz qabuliga yozib qo'yaymi?" deb javob bering.
- TIL VA SHEVALAR: Mijoz qaysi tilda (o'zbekcha adabiy, shevada, lotin, kirill yoki ruscha) yozsa, o'sha tilda muloyim va xushmuomala javob bering.
- AGAR BAZADA JAVOB BO'LMASA: "Assalomu alaykum! Men klinikamiz administratoriman. Ushbu savolingizga to'liqroq aniqlik kiritishim uchun iltimos, ismingiz va telefon raqamingizni qoldiring, siz bilan bog'lanib qabulga yozib qo'yaman!" deb javob bering.

Suhbatdoshga har doim hurmat bilan "Siz" deb murojaat qiling. Javobingiz qisqa, tushunarli va samimiy bo'lsin.
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
        """Provides realistic administrator response in dev mode when OpenAI API key is missing."""
        if kb_context:
            best_match = kb_context[0]
            return (
                f"{best_match['answer']}\n\n"
                "Sizga qaysi kun va vaqt qulay? Sizni shifokorimiz qabuliga yozib qo'yaymi?"
            )
        return (
            "Assalomu alaykum! Men klinikamiz administratoriman. Savolingiz bo'yicha sizga to'liqroq "
            "ma'lumot berishim uchun iltimos ismingiz va telefon raqamingizni qoldiring, sizni shifokorimiz qabuliga yozib qo'yaman!"
        )
