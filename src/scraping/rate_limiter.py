"""Client-side rate limiting with random jitter."""
import random
import threading
import time


class RateLimiter:
    """Throttle requests to at most one every ``min_interval`` seconds.

    A random jitter (``min_interval * jitter`` spread) is added so requests do
    not fire in a clockwork rhythm that anti-bot systems can fingerprint.
    """

    def __init__(
        self,
        min_interval: float = 1.2,
        max_interval: float = 2.4,
        jitter: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.min_interval = min_interval
        self.max_interval = max(interval := max_interval, min_interval)
        self.jitter = jitter
        self._rng = random.Random(seed)
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def _sample_delay(self) -> float:
        base = self._rng.uniform(self.min_interval, self.max_interval)
        return max(0.0, base * self._rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))

    def wait(self) -> None:
        """Block until the next request is allowed."""
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_allowed - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._next_allowed = max(now, self._next_allowed) + self._sample_delay()

    def __enter__(self) -> "RateLimiter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
