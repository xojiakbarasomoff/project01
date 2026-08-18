import logging
from typing import List, Dict, Any, Optional

from app.core.clients import get_openai_client
from app.core.config import settings
from app.services.guardrails import GuardrailService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Siz stomatologiya klinikasining rasmiy klinika administratorisiz (resepshn). Siz klinika nomidan bemorlar bilan bevosita, o'ta muloyim, samimiy va professional muloqot qilasiz.

SIZNING ASOSIY MAQSADINGIZ VA MUAMMO YECHIMI:
1. Bemorlarning savollariga faqat va faqat taqdim etilgan "BILIM BAZASI KONTEKSTI" (.txt / .md hujjat) asosida aniq va xushmuomala javob berish.
2. Bemorning alifbosi va tilini ANIQ VA QAT'IY AKSLANTIRISH (MIRRORING):
   - Agar bemor Kirill alifbosida yozsa -> FAQAT KIRILL ALIFBOSIDA JAVOB BERING (Salomlashish: "Ассалому алайкум!").
   - Agar bemor Lotin alifbosida yozsa -> FAQAT LOTIN ALIFBOSIDA JAVOB BERING (Salomlashish: "Assalomu alaykum!").
   - Agar bemor Rus tilida yozsa -> FAQAT RUS TILIDA JAVOB BERING (Salomlashish: "Здравствуйте!").
3. AKTIV TINGLASH (ACTIVE LISTENING): Bemorning muammosiga (masalan, tish og'rig'i, estetik ehtiyoj, narx) hamdardlik va e'tibor ko'rsatib, samimiy muloqot qiling. Robotik quruq javoblar taqiqlanadi!
4. QULAY VAQT VA TELEFON RAQAMINI OLISH (LEAD CAPTURE):
   - Aniq narxlar bo'lsa aytasiz. Narxlar holatga ko'ra o'zgaruvchan bo'lsa, bepul shifokor ko'rigi zarurligini tushuntirib, bemorning telefon raqami va qo'ng'iroq qilish uchun QULAY VAQTINI (time-slot) so'rab oling.
5. MAVJUD BO'LMAGAN XIZMATLAR (NO HALLUCINATION):
   - Agar so'ralgan xizmat turi klinikaning bilimlar bazasida bo'lmasa, aniq va muloyim aytasiz: "Afsuski, bizda bunday xizmat turi hozircha yo'q" (yoki Kirill/Ruscha muqobili).
   - QAT'IYAN TAQIQLANADI: Boshqa klinikalarni yoki raqobatchilarni tavsiya qilish ("Bunday ma'lumot bizda afsuski yo'q" deb to'xtating).
6. TIBBIY CHEKLANISH: Retsept yoki dori-darmon tavsiya qilmaysiz. Har doim bepul shifokor ko'rigiga taklif eting.

Suhbatdoshga har doim hurmat bilan "Siz" deb murojaat qiling. Javobingiz samimiy va jonli bo'lsin.
"""


class LLMService:
    @classmethod
    async def generate_response(
        cls,
        user_message: str,
        kb_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        strict_rules: Optional[List[str]] = None
    ) -> str:
        """Generates AI response using GPT-4o mini / Gemini given user message, strict rules and RAG context."""

        # Construct Context block from RAG Knowledge Base / Document matches
        if kb_context:
            context_str = "\n".join([
                f"- [{item.get('category', 'doc')}] {item.get('question', '')}:\n  {item.get('answer', '')}"
                for item in kb_context
            ])
        else:
            context_str = "Bilim bazasida mos keladigan ma'lumot topilmadi."

        script = GuardrailService.detect_script(user_message)
        lang = GuardrailService.detect_language(user_message)
        greeting = GuardrailService.get_greeting(user_message)

        system_blocks = []

        if strict_rules:
            rules_text = "\n".join([f"- {r.strip()}" for r in strict_rules if r.strip()])
            if rules_text:
                strict_block = (
                    "=================== ⚠️ QAT'IY QOIDALAR (STRICT GUARDRAILS) ===================\n"
                    "SIZ USHBU QOIDALARGA HAR BIR JAVOBINGIZDA 100% QAT'IY AMAL QILISHINGIZ SHART!\n"
                    "SHU QOIDALARDAN CHETGA CHIQISH QAT'IYAN TAQIQLANADI. AGAR USHBU QOIDALAR BOSHQALAR BILAN ZID KELSA, USHBU QOIDALAR HAR DOIM BIRINCHI O'RINDA TURADI:\n\n"
                    f"{rules_text}\n"
                    "================================================================================"
                )
                system_blocks.append(strict_block)

        system_blocks.append(SYSTEM_PROMPT)

        system_blocks.append(
            f"Foydalanuvchi tili/scripti: lang={lang}, script={script}. Tavsiya etilgan salomlashish: {greeting}\n\n"
            f"=== BILIM BAZASI KONTEKSTI ===\n{context_str}\n==============================="
        )

        full_system_prompt = "\n\n".join(system_blocks)

        prompt_messages = [
            {"role": "system", "content": full_system_prompt}
        ]

        if conversation_history:
            for msg in conversation_history[-6:]:
                prompt_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        prompt_messages.append({"role": "user", "content": user_message})

        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-fake"):
            return cls._dev_fallback_response(user_message, kb_context)

        try:
            client = get_openai_client()
            model_name = settings.OPENAI_MODEL
            if (
                settings.OPENAI_API_KEY.startswith("AQ.")
                or settings.OPENAI_API_KEY.startswith("AIza")
            ) and model_name == "gpt-4o-mini":
                model_name = "gemini-3.5-flash-lite"

            temperature_val = 0.0 if strict_rules else 0.2

            response = await client.chat.completions.create(
                model=model_name,
                messages=prompt_messages,
                temperature=temperature_val,
                max_tokens=450
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI Chat Completion: {str(e)}")
            return cls._dev_fallback_response(user_message, kb_context)

    @staticmethod
    def _dev_fallback_response(user_message: str, kb_context: List[Dict[str, Any]]) -> str:
        """Provides realistic administrator response in dev mode matching script and language."""
        script = GuardrailService.detect_script(user_message)
        lang = GuardrailService.detect_language(user_message)
        greeting = GuardrailService.get_greeting(user_message)

        msg_lower = user_message.lower()

        booking_confirm_keywords = ["ha", "yozing", "yozib", "ism", "tel", "raqam", "+998", "soat", "ertaga", "bugun", "bormoqchiman", "да", "номер", "запишите"]
        has_number = any(char.isdigit() for char in user_message)

        if any(kw in msg_lower for kw in booking_confirm_keywords) or has_number:
            if lang == "ru":
                return (
                    f"{greeting} Большое спасибо! Я приняла ваши данные и передала их нашему старшему администратору. "
                    "Мы свяжемся с вами в удобное для вас время для подтверждения записи на прием!"
                )
            elif script == "cyrillic":
                return (
                    f"{greeting} Катта раҳмат! Маълумотларингизни қабул қилдим ва катта администраторимизga узатдим! "
                    "Сиз кўрсатган вақтда сиз билан боғланиб, шифокор қабулини тасдиқлаймиз!"
                )
            else:
                return (
                    f"{greeting} Katta rahmat! Ma'lumotlaringizni qabul qildim va administratorimizga uzatdim! "
                    "Siz ko'rsatgan qulay vaqtda siz bilan bog'lanib, shifokor qabulini tasdiqlaymiz!"
                )

        if kb_context:
            best_match = kb_context[0]
            answer = best_match['answer']
            if lang == "ru":
                return f"{greeting} {answer}\n\nВ какое время вам удобно подойти на бесплатный осмотр? Оставьте ваш номер телефона и удобное время."
            elif script == "cyrillic":
                return f"{greeting} {answer}\n\nСизга қайси кун ва вақтда бепул кўрикга келиш қулай? Telefon рақамингиз ва қулай вақтни қолдиринг."
            else:
                return f"{greeting} {answer}\n\nSizga qaysi kun va soatda bepul ko'rikka kelish qulay? Telefon raqamingiz va qulay vaqtingizni qoldiring."

        if lang == "ru":
            return f"{greeting} К сожалению, у нас пока нет такой услуги. Уточните, пожалуйста, ваш номер телефона и удобное время для звонка, чтобы наш врач провел консультацию!"
        elif script == "cyrillic":
            return f"{greeting} Афсуски, бизда бундай хизмат тури ҳозирча йўқ. Агар бошқа саволларингиз бўлса, телефон рақамингиз ва қулай вақтингизни қолдиринг!"
        return f"{greeting} Afsuski, bizda bunday xizmat turi hozircha yo'q. Agar boshqa savollaringiz bo'lsa, telefon raqamingiz va qulay vaqtingizni qoldiring!"

    @classmethod
    async def extract_lead_topic(
        cls,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Extracts a concise primary intent/topic (max 10-12 words) from user inquiry for lead tracking.
        Does NOT return full chat context, only the concise intent.
        """
        prompt = (
            "Quyidagi bemor murojaatidan uning ASOSIY MAQSADINI 1 ta qisqa ibora bilan ajratib ber (maksimum 10 so'z).\n"
            "Misollar: 'Tish plomba qildirish narxi', 'Implantatsiya bo'yicha konsultatsiya', 'Tish og'rig'i davolash', 'Shifokor qabuliga yozilish'.\n"
            "To'liq chat matnini yozma, FAQAT bemorning qisqa va londa maqsadini yoz.\n\n"
            f"Murojaat matni: {user_message}"
        )
        try:
            res = await cls.generate_response(
                user_message=prompt,
                kb_context=[],
                conversation_history=None,
                strict_rules=["Faqat 1-2 iboradan iborat londa maqsadni yoz. Ortiqcha izoh yoki gap yozmang."]
            )
            clean = res.strip().strip('"').strip("'").strip(".")
            if len(clean) > 150:
                clean = clean[:147] + "..."
            return clean
        except Exception as e:
            logger.warning(f"Failed to extract lead topic via LLM: {e}")
            return user_message[:100]


