from typing import Any, Optional

from braandket.backend import Backend
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

    # tensor constructors

    def zeros(self, *, backend: Optional[Backend] = None):
        from braandket.tensor import zeros
        return zeros(self, backend=backend)

    def ones(self, *, backend: Optional[Backend] = None):
        from braandket.tensor import ones
        return ones(self, backend=backend)

    def full(self, value: Any, *, backend: Optional[Backend] = None):
        return self.ones(backend=backend) * value

    def values(self, values: Any, *, backend: Optional[Backend] = None):
        from braandket.tensor import NumericTensor
        return NumericTensor.of(values, [self], backend=backend)
