import abc
import warnings
from typing import Optional

from braandket.backend import Backend
from .space import Space


class HSpace(Space, abc.ABC):
    @property
    @abc.abstractmethod
    def ct(self) -> 'HSpace':
        pass

    @property
    @abc.abstractmethod
    def ket(self) -> 'KetSpace':
        pass

    @property
    @abc.abstractmethod
    def bra(self) -> 'BraSpace':
        pass


class KetSpace(HSpace):
    def __init__(self, n: int, name: Optional[str] = None):
        super().__init__()
        self._n = n
        self._name = name

        self._bra = BraSpace(self)

    # basics

    @property
    def n(self) -> int:
        return self._n

    @property
    def name(self) -> Optional[str]:
        return self._name

    # spaces

    @property
    def ct(self) -> 'BraSpace':
        return self._bra

    @property
    def ket(self) -> 'KetSpace':
        return self

    @property
    def bra(self) -> 'BraSpace':
        return self._bra

    # tensor constructors

    def eigenstate(self, index: int, *, backend: Optional[Backend] = None):
        from braandket.tensor import eigenstate
        return eigenstate(self, index, backend=backend)

    def operator(self, ket_index: int, bra_index: int, *, backend: Optional[Backend] = None):
        from braandket.tensor import operator
        return operator(self, ket_index, bra_index, backend=backend)

    def projector(self, index: int, *, backend: Optional[Backend] = None):
        from braandket.tensor import projector
        return projector(self, index, backend=backend)

    def identity(self, *, backend: Optional[Backend] = None):
        from braandket.tensor import identity
        return identity(self, backend=backend)


class BraSpace(HSpace):
    def __new__(cls, ket: KetSpace):
        try:
            bra = ket.bra
            warnings.warn(
                "Please avoid creating a BraSpace yourself. "
                "Instead, use property \".ct\" on KetSpace instance.",
                category=UserWarning, stacklevel=2)
            return bra
        except AttributeError:
            return super().__new__(cls)

    def __init__(self, ket: KetSpace):
        super().__init__()

        if not isinstance(ket, KetSpace):
            raise TypeError("The parameter ket must be a KetSpace instance!")

        self._ket = ket

    # basics

    @property
    def n(self) -> int:
        return self._ket.n

    @property
    def name(self) -> Optional[str]:
        return self._ket.name

    # spaces

    @property
    def ct(self) -> KetSpace:
        return self._ket

    @property
    def ket(self) -> KetSpace:
        return self._ket

    @property
    def bra(self) -> 'BraSpace':
        return self
