import logging
from typing import List, Dict, Any, Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clients import get_openai_client
from app.core.config import settings
from app.models.domain import KnowledgeBase

logger = logging.getLogger(__name__)


class RAGService:
    @staticmethod
    async def get_embedding(text_content: str) -> Optional[List[float]]:
        """Generates 1536-dimensional embedding vector for text using OpenAI API."""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-fake"):
            # Return dummy 1536-dim normalized vector for offline testing
            import math
            dummy = [0.01 * (i % 100) for i in range(1536)]
            norm = math.sqrt(sum(x * x for x in dummy))
            return [x / norm for x in dummy]

        try:
            client = get_openai_client()  # singleton — no new connection pool per call
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
        Performs semantic vector search against tenant's knowledge_base in pgvector.
        Fallback to SQL ILIKE matching if vector search returns no results or embedding fails.
        """
        embedding = await cls.get_embedding(query)
        results: List[Dict[str, Any]] = []

        if embedding:
            try:
                # Perform pgvector cosine distance search
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

        # Fallback / Complementary text matching if vector search yields < top_k results
        if len(results) < top_k:
            query_words = [w for w in query.lower().split() if len(w) > 3]
            if query_words:
                # Push ILIKE filtering to the database — no full table scan in Python
                ilike_conditions = [
                    or_(
                        KnowledgeBase.question.ilike(f"%{w}%"),
                        KnowledgeBase.answer.ilike(f"%{w}%")
                    )
                    for w in query_words
                ]
                stmt = (
                    select(KnowledgeBase)
                    .where(
                        KnowledgeBase.tenant_id == tenant_id,
                        KnowledgeBase.is_active.is_(True),
                        or_(*ilike_conditions)
                    )
                    .limit(top_k)
                )
                res = await session.execute(stmt)
                all_kb = res.scalars().all()

                existing_questions = {r["question"] for r in results}
                for kb in all_kb:
                    if kb.question not in existing_questions:
                        results.append({
                            "question": kb.question,
                            "answer": kb.answer,
                            "category": kb.category
                        })
                        existing_questions.add(kb.question)
                        if len(results) >= top_k:
                            break

        return results
