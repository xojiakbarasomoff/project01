import json
import logging
import time
from typing import List, Dict, Any, Optional

from app.core.clients import get_arq_pool, get_redis_client

logger = logging.getLogger(__name__)

# ── Lua script for atomic debounce check-and-clear ───────────────────────────
# Atomically:
#   1. Read the current timer value.
#   2. If it's newer than job_timestamp (by > 1 ms) → return nil (debounced).
#   3. Otherwise → LRANGE + DEL list & timer keys and return the messages.
#
# All three Redis operations execute inside a single server-side transaction so
# there is no TOCTOU race between concurrent worker jobs.

_DEBOUNCE_LUA = """
local timer_key  = KEYS[1]
local list_key   = KEYS[2]
local job_ts     = tonumber(ARGV[1])

local latest = tonumber(redis.call('GET', timer_key))

if latest and latest > job_ts + 0.001 then
    return nil   -- newer message exists; let that job handle it
end

local msgs = redis.call('LRANGE', list_key, 0, -1)
redis.call('DEL', list_key)
redis.call('DEL', timer_key)
return msgs
"""


class DebounceService:
    @classmethod
    async def add_message_and_debounce(
        cls,
        tenant_id: int,
        user_id: int,
        external_chat_id: str,
        message_text: str,
        channel_type: str = "telegram",
        debounce_seconds: int = 30,
        business_connection_id: Optional[str] = None
    ) -> bool:
        """
        Appends message to Redis pending list and resets the TTL debounce timer.
        Schedules an ARQ worker job with a delay equal to debounce_seconds.
        Uses the process-wide shared Redis and ARQ pool — no new connections per call.
        """
        r = await get_redis_client()
        list_key = f"pending_messages:{tenant_id}:{user_id}"
        timer_key = f"debounce_timer:{tenant_id}:{user_id}"
        timestamp = time.time()

        msg_payload = {
            "text": message_text,
            "timestamp": timestamp,
            "external_chat_id": external_chat_id,
            "channel": channel_type,
            "business_connection_id": business_connection_id
        }

        try:
            # 1. Append message to Redis pending list
            await r.rpush(list_key, json.dumps(msg_payload))

            # 2. Update timer key with current timestamp and set TTL
            await r.set(timer_key, str(timestamp), ex=debounce_seconds + 5)

            # 3. Enqueue ARQ delayed worker job via shared pool
            arq_pool = await get_arq_pool()
            await arq_pool.enqueue_job(
                "process_debounce_batch",
                tenant_id,
                user_id,
                external_chat_id,
                timestamp,
                _defer_by=debounce_seconds
            )
            return True
        except Exception as e:
            logger.error(f"Error in add_message_and_debounce: {str(e)}")
            return False

    @classmethod
    async def get_and_clear_batched_messages(
        cls,
        tenant_id: int,
        user_id: int,
        job_timestamp: float
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Atomically checks whether this is still the latest job for the user and,
        if so, pops all pending messages from Redis in a single Lua transaction.

        Returns:
          - ``None``  if a newer message arrived after this job was scheduled
                      (debounced — the newer job will handle it).
          - ``[]``    if the list is empty (already processed by another job).
          - ``[...]`` the list of batched messages to process.
        """
        r = await get_redis_client()
        timer_key = f"debounce_timer:{tenant_id}:{user_id}"
        list_key = f"pending_messages:{tenant_id}:{user_id}"

        try:
            # Execute the Lua script atomically — eliminates TOCTOU race conditions
            raw_result = await r.eval(
                _DEBOUNCE_LUA,
                2,                       # number of KEYS
                timer_key,
                list_key,
                str(job_timestamp)       # ARGV[1]
            )

            if raw_result is None:
                logger.info(
                    f"⏳ Debouncing: newer message received for user {user_id}. Postponing."
                )
                return None

            if not raw_result:
                return []

            messages = [json.loads(m) for m in raw_result]
            return messages
        except Exception as e:
            logger.error(f"Error retrieving batched messages: {str(e)}")
            return []
