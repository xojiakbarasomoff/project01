import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.bot.runner import run_bot_polling

if __name__ == "__main__":
    asyncio.run(run_bot_polling())
