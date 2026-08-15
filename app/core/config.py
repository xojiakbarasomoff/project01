from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Medical Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000

    # ── Secrets (no defaults — pydantic-settings will raise on startup if missing) ──
    ENCRYPTION_KEY: str
    POSTGRES_PASSWORD: str
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_WEBHOOK_URL: str = ""
    TELEGRAM_WEBHOOK_SECRET: str = ""  # Used to verify X-Telegram-Bot-Api-Secret-Token

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── Database ──
    POSTGRES_SERVER: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "aimed_db"
    POSTGRES_USER: str = "postgres"

    # DATABASE_URL is assembled from the individual parts; can be overridden via env.
    DATABASE_URL: str = ""

    # ── Redis ──
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://redis:6379/0"

    # ── CORS ──
    # Comma-separated list of allowed origins. Example: "https://app.example.com,https://admin.example.com"
    CORS_ORIGINS: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def assemble_db_url(self) -> "Settings":
        """Build DATABASE_URL from parts if not explicitly set."""
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return self

    def get_cors_origins(self) -> list[str]:
        """Return list of allowed CORS origins. Falls back to localhost for development."""
        if self.CORS_ORIGINS:
            return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        if self.APP_ENV == "development":
            return ["http://localhost:3000", "http://localhost:8000", "http://localhost:8001"]
        return []


settings = Settings()
