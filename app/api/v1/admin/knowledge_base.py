import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import KnowledgeBase
from app.schemas.knowledge_base import KBCreate, KBUpdate, KBResponse
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/knowledge-base", tags=["Admin — Knowledge Base (FAQ)"])


@router.get("", response_model=List[KBResponse])
async def list_knowledge_base(
    tenant_id: int = Query(1, description="Tenant ID"),
    category: Optional[str] = Query(None, description="Category filter"),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves FAQ entries for a clinic tenant."""
    stmt = select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant_id)
    if category:
        stmt = stmt.where(KnowledgeBase.category == category)

    stmt = stmt.order_by(desc(KnowledgeBase.id))
    res = await db.execute(stmt)
    return res.scalars().all()


@router.post("", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
async def create_kb_item(
    payload: KBCreate,
    db: AsyncSession = Depends(get_db)
):
    """Adds a new FAQ entry and generates its pgvector embedding."""
    # Generate embedding
    text_to_embed = f"{payload.question} {payload.answer}"
    embedding = await RAGService.get_embedding(text_to_embed)

    kb = KnowledgeBase(
        tenant_id=payload.tenant_id,
        question=payload.question,
        answer=payload.answer,
        category=payload.category,
        embedding=embedding,
        is_active=payload.is_active
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.put("/{kb_id}", response_model=KBResponse)
async def update_kb_item(
    kb_id: int,
    payload: KBUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Updates an existing FAQ entry and re-indexes its embedding."""
    stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    res = await db.execute(stmt)
    kb = res.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="FAQ item not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(kb, field, val)

    # Re-generate vector embedding
    text_to_embed = f"{kb.question} {kb.answer}"
    kb.embedding = await RAGService.get_embedding(text_to_embed)

    await db.commit()
    await db.refresh(kb)
    return kb


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb_item(
    kb_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Deletes an FAQ entry."""
    stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    res = await db.execute(stmt)
    kb = res.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="FAQ item not found")

    await db.delete(kb)
    await db.commit()
    return None
