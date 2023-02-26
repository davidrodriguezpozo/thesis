from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Literal, Optional

from _types.results import BaseResults


class log_level(Enum):
    verbose = auto()
    info = auto()
    error = auto()


class Logger:
    def __init__(self, level: log_level = log_level.info):
        self.level = level

    @contextmanager
    def log_message(self, func: Callable[..., Any], *args, **kwargs) -> None:
        start = timer()
        if self.level.value < log_level.error.value:
            print(f"[{start}] - Starting {func.__name__}")

        if self.level.value == log_level.verbose.value:
            print(f"Arguments [{args}] [{kwargs}]")

        # ----
        yield func
        # ----

        if self.level.value < log_level.error.value:
            now = timer()
            print(f"[{timer()}] - Ended {func.__name__}. Elapsed: {now - start}")

    def log(self, func: Callable[..., Any]):
        @wraps(func)
        def log_method(*args, **kwargs):
            with self.log_message(func, *args, **kwargs):
                res = func(*args, **kwargs)
            return res

        return log_method

    def print_results(
        self, sorted_results: Dict[str, List[BaseResults]], code: Optional[str] = None
    ) -> None:
        if not sorted_results:
            return
        print("")
        print(" ------ RESULTS ------ ".center(40))
        if code:
            print(f" - Study: {code} - ".center(40))
        print("")
        print("Top results: ")
        top_results = {k: v[0] for k, v in sorted_results.items()}
        for res in top_results:
            print(f"Top result for {res}: {top_results[res].accuracy}")

        top_result = sorted(
            top_results.values(), key=lambda x: x.accuracy, reverse=True
        )[0]
        print(
            f"Best algorithm: {top_result.method} with an ccuracy of {top_result.accuracy}."
        )
        top_result.plot()


logger = Logger()
