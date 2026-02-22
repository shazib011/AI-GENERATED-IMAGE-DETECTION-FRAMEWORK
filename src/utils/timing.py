import time
from contextlib import contextmanager

@contextmanager
def timer_ms():
    t0 = time.time()
    yield lambda: int((time.time() - t0) * 1000)
