from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.domain import Operator
from app.core.security import hash_password, verify_password, create_admin_token, verify_admin_token

router = APIRouter(prefix="/auth", tags=["Admin Auth"])


class LoginSchema(BaseModel):
    username: str
    password: str


class ChangePasswordSchema(BaseModel):
    old_password: str
    new_password: str


async def get_current_admin(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> str:
    """Dependency to verify admin access token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autentifikatsiya talab etiladi"
        )
    token = authorization.replace("Bearer ", "").strip()
    username = verify_admin_token(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yaroqsiz yoki muddati o'tgan token"
        )
    return username


@router.post("/login")
async def login(data: LoginSchema, db: AsyncSession = Depends(get_db)):
    """Authenticate admin operator and return access token."""
    username = data.username.strip()
    password = data.password.strip()

    res = await db.execute(select(Operator).where(Operator.name == username))
    operator = res.scalar_one_or_none()

    # Auto-seed default admin operator if database is empty
    if not operator:
        res_any = await db.execute(select(Operator))
        any_op = res_any.first()
        if not any_op and username == "admin":
            operator = Operator(
                tenant_id=1,
                name="admin",
                role="admin",
                credentials=hash_password("admin")
            )
            db.add(operator)
            await db.commit()
            await db.refresh(operator)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login yoki parol noto'g'ri"
            )

    if not verify_password(password, operator.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login yoki parol noto'g'ri"
        )

    token = create_admin_token(operator.name)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": operator.name
    }


@router.get("/me")
async def get_me(username: str = Depends(get_current_admin)):
    """Check current auth status."""
    return {"authenticated": True, "username": username}


@router.post("/change-password")
async def change_password(
    data: ChangePasswordSchema,
    username: str = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Change password for logged-in operator."""
    res = await db.execute(select(Operator).where(Operator.name == username))
    operator = res.scalar_one_or_none()
    if not operator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Foydalanuvchi topilmadi"
        )

    if not verify_password(data.old_password, operator.credentials):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Joriy (eski) parol noto'g'ri kiritildi"
        )

    if len(data.new_password) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yangi parol kamida 4 ta belgidan iborat bo'lishi kerak"
        )

    operator.credentials = hash_password(data.new_password)
    await db.commit()
    return {"status": "success", "message": "Parol muvaffaqiyatli o'zgartirildi"}
