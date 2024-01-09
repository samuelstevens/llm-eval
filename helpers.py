import functools
import time


def timed(f):
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        t1 = time.perf_counter()
        return result, (t1 - t0)

    return wrap
