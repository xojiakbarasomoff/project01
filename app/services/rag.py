import logging
from typing import List, Dict, Any, Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clients import get_openai_client
from app.core.config import settings
from app.models.domain import KnowledgeBase, Tenant

logger = logging.getLogger(__name__)

DEFAULT_CLINIC_DOC = """=== SHIFO SHAMSI STOMATOLOGIYA KLINIKASI BILIMLAR BAZASI ===

1. KLINIKA HAQIDA MA'LUMOT:
- Klinika nomi: "Shifo Shamsi" Stomatologiya Markazi
- Manzil: Toshkent shahri, Chilonzor tumani, 9-mavze, 24-uy (Metro: Chilonzor bekatidan 5 daqiqalik yo'l)
- Mo'ljal: Rayhon milliy taomlari qarshisida
- Telefon: +998 (71) 200-00-99, +998 (90) 123-45-67
- Ish vaqti: Dushanba - Shanba: 09:00 dan 20:00 gacha. Yakshanba: 10:00 dan 16:00 gacha.

2. SHIFOKORLARIMIZ VA MUTAXASSISLIKLAR:
- Dr. Jasur Alimov — Bosh shifokor, Implantolog-xirurg (15 yillik tajriba)
- Dr. Malika Rustamova — Stomatolog-terapevt, Estetik restavratsiya bo'yicha mutaxassis
- Dr. Bekzod Shodmonov — Ortodont (Breket va elaynerlar bo'yicha mutaxassis)
- Dr. Nigora Ahmedova — Bolalar stomatologi (Pediatric dentist)

3. XIZMATLAR VA NARXLAR:
- Bepul konsultatsiya va ko'rik: Har bir bemor uchun birinchi ko'rik va rentgen diagnostika BEPUL.
- Tish kavagini davolash va plomba: 150 000 - 350 000 so'm (kavaki chuqurligiga qarab).
- Tish ildizi (kanal) davolash (Endodontiya): 250 000 - 500 000 so'm.
- Professional tish tozalash va oqartirish (AirFlow + Ultratovush): 300 000 - 600 000 so'm.
- Metal-keramika koronka (Toj): 450 000 - 800 000 so'm.
- Sirkoniyni koronka (Premium): 1 200 000 - 1 800 000 so'm.
- Tish implantatsiyasi (Osstem / Straumann, Koreya va Shveysariya): 2 500 000 - 5 000 000 so'm (Implantatsiya narxi individual ravishda bepul ko'rikda aniqlanadi).
- Breket tizimi (Metal / Keramika): 3 000 000 - 7 000 000 so'm (Har ikkala jag' uchun).
- Vinirlar (E-max estetik vinirlar): 1 500 000 - 2 500 000 so'm.
- Tish oldirish (Xirurgiya): 150 000 - 400 000 so'm. Operatsion oldirish (Aql tishi): 400 000 - 800 000 so'm.

4. MUHIM QOIDALAR:
- Narxlar holatga ko'ra o'zgarishi mumkin bo'lgan taqdirda, bemorga bepul ko'rik va shifokor ko'rigi taklif etiladi va uning telefon raqami hamda qulay vaqti so'rab olinadi.
- Dori-darmon yoki retsept berish taqiqlangan, bemor har doim shifokor ko'rigiga taklif qilinadi.
- Agar so'ralgan xizmat ushbu ro'yxatda bo'lmasa: "Afsuski, bizda bunday xizmat turi hozircha yo'q" deb javob beriladi. Boshqa klinikalarga yo'naltirilmaydi.
"""

DEFAULT_STRICT_RULES = [
    "1. RETSEPT VA DORI TAQIQLANADI: Dori-darmon yoki tibbiy retseptlar tavsiya qilish qat'iyan man etiladi.",
    "2. TASHXIS QO'YISH TAQIQLANADI: Masofadan tashxis qo'yish taqiqlanadi. Bemor har doim shifokor ko'rigiga yo'naltirilishi shart.",
    "3. FAQAT KLINIKA SAVOLLARI: Faqat va faqat klinika faoliyati, xizmatlari, narxlari va shifokorlariga bog'liq savollarga javob berilsin. Siyosiy, shaxsiy yoki aloqasiz savollarga javob berilmaysiz.",
    "4. BOSHQA KLINIKALAR TAQIQLANADI: Boshqa klinikalarni yoki raqobatchilarni tavsiya qilish qat'iyan taqiqlanadi.",
    "5. LEAD CAPTURE: Bemor bilan muloqotda uning ismi, telefon raqami hamda qabulga kelish uchun qulay vaqtini olishga intiling."
]


class RAGService:
    @staticmethod
    async def get_clinic_doc(session: AsyncSession, tenant_id: int) -> str:
        """Retrieves tenant's clinic knowledge document (.txt / .md). Defaults to DEFAULT_CLINIC_DOC if not set."""
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        res = await session.execute(stmt)
        tenant = res.scalar_one_or_none()
        if tenant and tenant.settings and isinstance(tenant.settings, dict):
            doc_text = tenant.settings.get("clinic_doc_text")
            if doc_text and doc_text.strip():
                return doc_text.strip()
        return DEFAULT_CLINIC_DOC.strip()

    @staticmethod
    async def update_clinic_doc(session: AsyncSession, tenant_id: int, doc_text: str) -> str:
        """Updates tenant's clinic knowledge document in settings JSONB."""
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        res = await session.execute(stmt)
        tenant = res.scalar_one_or_none()
        if tenant:
            current_settings = dict(tenant.settings or {})
            current_settings["clinic_doc_text"] = doc_text.strip()
            tenant.settings = current_settings
            await session.commit()
            return doc_text.strip()
        return doc_text.strip()

    @staticmethod
    async def get_strict_rules(session: AsyncSession, tenant_id: int) -> List[str]:
        """Retrieves tenant's strict guardrail rules. Defaults to DEFAULT_STRICT_RULES if not set."""
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        res = await session.execute(stmt)
        tenant = res.scalar_one_or_none()
        if tenant and tenant.settings and isinstance(tenant.settings, dict):
            rules = tenant.settings.get("strict_rules")
            if isinstance(rules, list) and len(rules) > 0:
                return rules
        return DEFAULT_STRICT_RULES

    @staticmethod
    async def update_strict_rules(session: AsyncSession, tenant_id: int, rules: List[str]) -> List[str]:
        """Updates tenant's strict guardrail rules in settings JSONB."""
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        res = await session.execute(stmt)
        tenant = res.scalar_one_or_none()
        if tenant:
            current_settings = dict(tenant.settings or {})
            clean_rules = [r.strip() for r in rules if r.strip()]
            current_settings["strict_rules"] = clean_rules
            tenant.settings = current_settings
            await session.commit()
            return clean_rules
        return rules

    @staticmethod
    async def get_embedding(text_content: str) -> Optional[List[float]]:
        """Generates 1536-dimensional embedding vector for text using OpenAI API."""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-fake"):
            import math
            dummy = [0.01 * (i % 100) for i in range(1536)]
            norm = math.sqrt(sum(x * x for x in dummy))
            return [x / norm for x in dummy]

        try:
            client = get_openai_client()
            emb_model = settings.EMBEDDING_MODEL
            if (
                settings.OPENAI_API_KEY.startswith("AQ.")
                or settings.OPENAI_API_KEY.startswith("AIza")
            ) and emb_model == "text-embedding-3-small":
                emb_model = "gemini-embedding-2"

            response = await client.embeddings.create(
                model=emb_model,
                input=text_content.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            return None

    @classmethod
    async def search_knowledge_base(
        cls,
        session: AsyncSession,
        tenant_id: int,
        query: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search & text search against clinic knowledge doc and pgvector.
        Returns matching knowledge items and relevant document sections.
        """
        results: List[Dict[str, Any]] = []

        # 1. Include clinic .txt/.md document content as primary context
        clinic_doc = await cls.get_clinic_doc(session, tenant_id)
        if clinic_doc:
            # Chunk document by sections or paragraphs
            sections = [s.strip() for s in clinic_doc.split("\n\n") if s.strip()]
            query_lower = query.lower()
            matching_sections = []
            for sec in sections:
                # Find matching sections by keywords
                words = [w for w in query_lower.split() if len(w) > 3]
                if any(w in sec.lower() for w in words):
                    matching_sections.append(sec)

            # If matching sections found, use them; otherwise attach main sections
            doc_context = "\n---\n".join(matching_sections[:3]) if matching_sections else clinic_doc[:1500]
            results.append({
                "question": "Klinika Bilimlar Bazasi Hujjati (.txt / .md)",
                "answer": doc_context,
                "category": "clinic_doc"
            })

        # 2. Search structured database KnowledgeBase items if available
        embedding = await cls.get_embedding(query)
        if embedding:
            try:
                stmt = (
                    select(KnowledgeBase)
                    .where(
                        KnowledgeBase.tenant_id == tenant_id,
                        KnowledgeBase.is_active.is_(True),
                        KnowledgeBase.embedding.isnot(None)
                    )
                    .order_by(KnowledgeBase.embedding.cosine_distance(embedding))
                    .limit(top_k)
                )
                res = await session.execute(stmt)
                items = res.scalars().all()
                for item in items:
                    results.append({
                        "question": item.question,
                        "answer": item.answer,
                        "category": item.category
                    })
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text match: {str(e)}")

        return results

