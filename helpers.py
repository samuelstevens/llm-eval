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


def fs_safe(s: str) -> str:
    """
    Makes a string safe for filesystems by replacing bad characters.
    """
    unsafe_chars = '<>:"/\\|?*'
    for ch in unsafe_chars:
        s = s.replace(ch, "_")
    return s
