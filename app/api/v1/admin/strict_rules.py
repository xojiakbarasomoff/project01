import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/strict-rules", tags=["Admin Strict Rules"])


class StrictRulesUpdateSchema(BaseModel):
    rules: List[str]


@router.get("", response_model=Dict[str, Any])
async def get_strict_rules(
    tenant_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve current strict guardrail rules for the tenant."""
    try:
        rules = await RAGService.get_strict_rules(db, tenant_id)
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "rules": rules
        }
    except Exception as e:
        logger.error(f"Error fetching strict rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch strict rules."
        )


@router.put("", response_model=Dict[str, Any])
async def update_strict_rules(
    payload: StrictRulesUpdateSchema,
    tenant_id: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Update strict guardrail rules for the tenant."""
    try:
        updated_rules = await RAGService.update_strict_rules(db, tenant_id, payload.rules)
        return {
            "status": "success",
            "message": "Qat'iy qoidalar muvaffaqiyatli saqlandi!",
            "rules": updated_rules
        }
    except Exception as e:
        logger.error(f"Error updating strict rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strict rules."
        )
