import asyncio
import httpx
import json
import time

BASE_URL = "http://localhost:8001"
TENANT_ID = 1


async def main():
    print("==================================================")
    print("🏥 AI Medical Assistant — Live Test Run")
    print("==================================================\n")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Health check
        resp = await client.get(f"{BASE_URL}/health")
        print(f"1. Service Health Check: {resp.json()}\n")

        # 2. Simulate Telegram patient sending rapid batched messages
        chat_id = 1234567
        print("2. Simulating incoming Telegram message batch from Patient (Jamshid)...")
        msg1 = {"update_id": 201, "message": {"message_id": 10, "from": {"id": chat_id, "first_name": "Jamshid"}, "chat": {"id": chat_id}, "text": "Assalomu alaykum, klinika manzilingiz qayerda?"}}
        msg2 = {"update_id": 202, "message": {"message_id": 11, "from": {"id": chat_id, "first_name": "Jamshid"}, "chat": {"id": chat_id}, "text": "Va tish plomba narxi qancha turadi?"}}

        r1 = await client.post(f"{BASE_URL}/api/v1/telegram/webhook/{TENANT_ID}", json=msg1)
        print(f"   Message 1 sent -> Response: {r1.json()}")

        r2 = await client.post(f"{BASE_URL}/api/v1/telegram/webhook/{TENANT_ID}", json=msg2)
        print(f"   Message 2 sent -> Response: {r2.json()}\n")

        print("3. Messages enqueued in Redis! Debounce batch window running...")
        print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
