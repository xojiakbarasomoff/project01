import logging
from sqlalchemy import text
from app.db.session import engine

logger = logging.getLogger(__name__)

async def init_db_schema():
    """Ensures dynamic tables like doctors exist on startup."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id SERIAL PRIMARY KEY,
                    tenant_id INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    specialty VARCHAR(255) NOT NULL DEFAULT 'Stomatolog',
                    phone VARCHAR(50),
                    working_hours VARCHAR(255) NOT NULL DEFAULT '09:00 - 18:00',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
                );
            """))

            res = await conn.execute(text("SELECT COUNT(*) FROM doctors;"))
            count = res.scalar()
            if count == 0:
                await conn.execute(text("""
                    INSERT INTO doctors (tenant_id, name, specialty, phone, working_hours) VALUES
                    (1, 'Dr. Sardor Rahimov', 'Stomatolog Shifokor', '+998 90 123 45 67', '09:00 - 18:00'),
                    (1, 'Dr. Malika Umarova', 'Ortodont Shifokor', '+998 91 987 65 43', '10:00 - 17:00'),
                    (1, 'Dr. Jasur Alimov', 'Implantolog Shifokor', '+998 93 555 44 33', '09:00 - 16:00'),
                    (1, 'Dr. Nigora Kimsanova', 'Bolalar Stomatologi', '+998 94 222 33 44', '09:00 - 15:00');
                """))
                logger.info("✅ Seeded default doctors in database.")
    except Exception as e:
        logger.error(f"Failed to init DB schema: {e}")
