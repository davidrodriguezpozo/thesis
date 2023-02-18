from typing import Protocol


class BaseResults(Protocol):
    method: str

    def __init__(self):
        pass

    @property
    def accuracy() -> float:
        ...

    def plot() -> None:
        ...
