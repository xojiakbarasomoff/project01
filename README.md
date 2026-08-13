# AI Medical Assistant — Klinikalar uchun sun'iy intellektli tibbiy operator

Multi-tenant AI Medical Assistant application built with Python 3.12, FastAPI, PostgreSQL (pgvector), Redis (Debounce/Queue), ARQ Worker, and OpenAI API (GPT-4o mini).

## Architecture & Technology Stack

- **Framework**: Python 3.12 + FastAPI (Async Web Framework)
- **Database**: PostgreSQL 16 with `pgvector` extension for semantic search (RAG)
- **Async Queue & Cache**: Redis 7 + ARQ for 15-40s message debouncing & background LLM workers
- **ORM & Migrations**: SQLAlchemy 2 (Async) + Alembic
- **Multi-Tenant Isolation**: Row-Level Tenant Isolation (`tenant_id`)
- **Security**: Sensitive token encryption at-rest via Fernet cryptography

## Quick Start (Docker Compose)

1. Clone the repository and copy environment variables:
   ```bash
   cp .env.example .env
   ```

2. Start all services via Docker Compose:
   ```bash
   docker compose up -d --build
   ```

3. Run database migrations:
   ```bash
   docker compose exec web alembic upgrade head
   ```

4. Seed the database with 100 Dental Clinic FAQs:
   ```bash
   docker compose exec web python scripts/seed_faq.py
   ```

5. Check API health:
   ```bash
   curl http://localhost:8000/health
   ```

## API Documentation

Interactive OpenAPI documentation is available once running at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
