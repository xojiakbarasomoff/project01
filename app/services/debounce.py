import json
import logging
import time
from typing import List, Dict, Any, Optional
import redis.asyncio as redis
from arq.connections import create_pool, RedisSettings
from app.core.config import settings

logger = logging.getLogger(__name__)


class DebounceService:
    @staticmethod
    async def get_redis_client() -> redis.Redis:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)

    @classmethod
    async def add_message_and_debounce(
        cls,
        tenant_id: int,
        user_id: int,
        external_chat_id: str,
        message_text: str,
        channel_type: str = "telegram",
        debounce_seconds: int = 30
    ) -> bool:
        """
        Appends message to Redis pending list and resets the TTL debounce timer.
        Schedules an ARQ worker job with a delay equal to debounce_seconds.
        """
        r = await cls.get_redis_client()
        list_key = f"pending_messages:{tenant_id}:{user_id}"
        timer_key = f"debounce_timer:{tenant_id}:{user_id}"
        timestamp = time.time()

        msg_payload = {
            "text": message_text,
            "timestamp": timestamp,
            "external_chat_id": external_chat_id,
            "channel": channel_type
        }

        try:
            # 1. Append message to Redis pending list
            await r.rpush(list_key, json.dumps(msg_payload))

            # 2. Update timer key with current timestamp and set TTL
            await r.set(timer_key, str(timestamp), ex=debounce_seconds + 5)

            # 3. Schedule ARQ delayed worker job
            arq_redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
            await arq_redis.enqueue_job(
                "process_debounce_batch",
                tenant_id,
                user_id,
                external_chat_id,
                timestamp,
                _defer_by=debounce_seconds
            )
            await arq_redis.close()
            return True
        except Exception as e:
            logger.error(f"Error in add_message_and_debounce: {str(e)}")
            return False
        finally:
            await r.close()

    @classmethod
    async def get_and_clear_batched_messages(
        cls,
        tenant_id: int,
        user_id: int,
        job_timestamp: float
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Checks if the latest message timer matches job_timestamp.
        If newer messages were added after this job was scheduled, returns None (debounced).
        Otherwise pops all pending messages from Redis list and returns them.
        """
        r = await cls.get_redis_client()
        list_key = f"pending_messages:{tenant_id}:{user_id}"
        timer_key = f"debounce_timer:{tenant_id}:{user_id}"

        try:
            latest_timer = await r.get(timer_key)

            # If a newer message was sent after this job, wait for the newer job to trigger
            if latest_timer and float(latest_timer) > job_timestamp + 0.001:
                logger.info(f"⏳ Debouncing: newer message received for user {user_id}. Postponing.")
                return None

            # Retrieve all batched messages
            raw_messages = await r.lrange(list_key, 0, -1)
            if not raw_messages:
                return []

            # Clear list and timer key
            await r.delete(list_key)
            await r.delete(timer_key)

            messages = [json.loads(m) for m in raw_messages]
            return messages
        except Exception as e:
            logger.error(f"Error retrieving batched messages: {str(e)}")
            return []
        finally:
            await r.close()
