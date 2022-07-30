import abc
import warnings
from typing import Optional

import numpy as np

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

    # tensors

    def eigenstate(self, index, *, sparse=True, dtype=np.float32):
        index = int(index)
        if not (0 <= index < self.n):
            raise ValueError(f"Illegal index: should be 0<=i<n, found i={index}, n={self.n}")
        if sparse:
            from ..tensor import SparseQTensor
            coordinate = (index,)
            value = np.asarray(True, dtype=dtype)
            return SparseQTensor([self], [(coordinate, value)])
        else:
            from ..tensor import NumpyQTensor
            values = np.zeros([self.n], dtype=dtype)
            values[index] = 1
            return NumpyQTensor([self], values)

    def identity(self, *, sparse=True, dtype=np.float32):
        if sparse:
            from ..tensor import SparseQTensor
            value = np.asarray(True, dtype=dtype)
            values = (((i, i), value) for i in range(self.n))
            return SparseQTensor([self, self.ct], values)
        else:
            from ..tensor import NumpyQTensor
            values = np.eye(self.n, dtype=dtype)
            return NumpyQTensor([self, self.ct], values)

    def operator(self, ket_index, bra_index, *, sparse=True, dtype=np.float32):
        return self.eigenstate(ket_index, sparse=sparse, dtype=dtype) @ \
               self.eigenstate(bra_index, sparse=sparse, dtype=dtype).ct

    def projector(self, index, *, sparse=True, dtype=np.float32):
        return self.operator(index, index, sparse=sparse, dtype=dtype)


class BraSpace(HSpace):
    def __new__(cls, ket: KetSpace):
        try:
            bra = ket.bra
            warnings.warn(
                "Please avoid creating a BraSpace yourself. "
                "To get a BraSpace, use .ct or .bra property on KetSpace instance.",
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
