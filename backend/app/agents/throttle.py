import threading
import time

from tenacity import retry, stop_after_attempt, wait_exponential

# Cohere trial keys allow 20 chat calls/minute; stay a little under it.
COHERE_MIN_INTERVAL_S = 3.5

_lock = threading.Lock()
_last_call = 0.0


def cohere_throttle():
    global _last_call
    with _lock:
        wait = _last_call + COHERE_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


llm_retry = retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5), reraise=True)
