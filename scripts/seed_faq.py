import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.domain import Tenant, Channel, KnowledgeBase
from app.core.security import encrypt_credentials
from app.core.config import settings


async def seed_data():
    async with AsyncSessionLocal() as session:
        print("🌱 Seeding initial database data...")

        # 1. Check or create default tenant
        stmt = select(Tenant).where(Tenant.id == 1)
        result = await session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant:
            tenant = Tenant(
                id=1,
                name="Stomatologiya Klinika #1",
                status="active",
                settings={
                    "debounce_seconds": 30,
                    "default_language": "uz",
                    "auto_escalate_attempts": 2
                }
            )
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)
            print(f"✅ Created default tenant: {tenant.name} (ID: {tenant.id})")
        else:
            print(f"ℹ️ Tenant already exists: {tenant.name} (ID: {tenant.id})")

        # 2. Check or create default Telegram channel
        stmt = select(Channel).where(Channel.tenant_id == tenant.id, Channel.type == "telegram")
        result = await session.execute(stmt)
        channel = result.scalar_one_or_none()

        credentials = {
            "bot_token": settings.TELEGRAM_BOT_TOKEN or "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ",
            "bot_username": "stomatologiya_ai_bot"
        }
        encrypted_creds = encrypt_credentials(credentials)

        if not channel:
            channel = Channel(
                tenant_id=tenant.id,
                type="telegram",
                credentials=encrypted_creds,
                is_active=True
            )
            session.add(channel)
            await session.commit()
            print("✅ Created default Telegram channel with encrypted credentials")
        else:
            channel.credentials = encrypted_creds
            await session.commit()
            print("🔄 Updated Telegram channel credentials in DB with current token")

        # 3. Load seed FAQs from json
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "dental_faq_seed.json")
        if not os.path.exists(json_path):
            print(f"⚠️ Seed JSON not found at {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)

        added_count = 0
        for item in faqs:
            stmt = select(KnowledgeBase).where(
                KnowledgeBase.tenant_id == tenant.id,
                KnowledgeBase.question == item["question"]
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if not existing:
                kb_item = KnowledgeBase(
                    tenant_id=tenant.id,
                    question=item["question"],
                    answer=item["answer"],
                    category=item.get("category", "general"),
                    is_active=True
                )
                session.add(kb_item)
                added_count += 1

        await session.commit()
        print(f"✅ Seeded {added_count} Knowledge Base FAQ entries for Tenant ID {tenant.id}")
        print("🚀 Database seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_data())
