from datetime import datetime, timezone

# A cap on new (uncached) answers per UTC day across all users. The per-IP rate limit stops one
# visitor from flooding the API, but many requests spread over a day could still use up the
# month's Groq spending limit in one day; this keeps the daily spend predictable. Cached answers
# do not count. The count lives in this process, so it also resets when the backend restarts.


class DailyLimit:
    def __init__(self, limit: int):
        self.limit = limit
        self.day = None
        self.used = 0

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def try_acquire(self) -> bool:
        today = self._today()
        if today != self.day:
            self.day, self.used = today, 0
        if self.limit > 0 and self.used >= self.limit:
            return False
        self.used += 1
        return True

    def release(self) -> None:
        # For a request that failed before producing an answer
        if self.used > 0:
            self.used -= 1
