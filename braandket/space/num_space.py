from typing import Optional

from .space import Space


class NumSpace(Space):
    def __init__(self, n: int, name: Optional[str] = None):
        self._n = n
        self._name = name

    # basics

    @property
    def n(self) -> int:
        return self._n

    @property
    def name(self) -> Optional[str]:
        return self._name
