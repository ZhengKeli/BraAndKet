import abc

import numpy as np


class Space(abc.ABC):
    @property
    @abc.abstractmethod
    def n(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def __repr__(self):
        name = hex(hash(self))[2:] if self.name is None else self.name
        return f"{type(self).__name__}(n={self.n},name={name})"

    @property
    def is_ket(self):
        return isinstance(self, KetSpace)

    @property
    def is_bra(self):
        return isinstance(self, BraSpace)

    @property
    def is_num(self):
        return isinstance(self, NumSpace)


class NumSpace(Space):
    def __init__(self, n, name=None):
        super().__init__()
        self._n = n
        self._name = name

    @property
    def n(self):
        return self._n

    @property
    def name(self):
        return self._name


class HSpace(Space, abc.ABC):
    @property
    @abc.abstractmethod
    def ct(self):
        pass

    @property
    @abc.abstractmethod
    def ket(self):
        pass

    @property
    @abc.abstractmethod
    def bra(self):
        pass

    @abc.abstractmethod
    def eigenstate(self, index, dtype=np.float32):
        pass

    @abc.abstractmethod
    def identity(self, dtype=np.float32):
        pass

    def projector(self, ket_index, bra_index=None, dtype=np.float32):
        if bra_index is None:
            bra_index = ket_index
        return self.ket.eigenstate(ket_index, dtype) @ self.bra.eigenstate(bra_index, dtype)

    def symmetry(self, index1, index2=None, dtype=np.float32):
        projector = self.projector(index1, index2, dtype)
        return projector + projector.ct


class KetSpace(HSpace):
    def __init__(self, n, name=None):
        super().__init__()
        self._n = n
        self._name = name

        self._eigenstates = np.eye(self.n, dtype=np.bool)

    @property
    def n(self):
        return self._n

    @property
    def name(self):
        return self._name

    @property
    def ct(self):
        return BraSpace(self)

    @property
    def ket(self):
        return self

    @property
    def bra(self):
        return self.ct

    def eigenstate(self, index, dtype=np.float32):
        from bnk.tensor import QTensor
        values = self._eigenstates[index]
        values = np.asarray(values, dtype=dtype)
        return QTensor([self], values)

    def identity(self, dtype=np.float32):
        from bnk.tensor import QTensor
        values = self._eigenstates
        values = np.asarray(values, dtype=dtype)
        return QTensor([self, self.ct], values)


class BraSpace(HSpace):
    def __init__(self, ket_space: KetSpace):
        super().__init__()

        if not isinstance(ket_space, KetSpace):
            raise TypeError("The parameter must be a KetSpace instance!")

        self._ket_space = ket_space

    @property
    def n(self):
        return self._ket_space.n

    @property
    def name(self):
        return self._ket_space.name

    @property
    def ct(self):
        return self._ket_space

    @property
    def ket(self):
        return self.ct

    @property
    def bra(self):
        return self

    def eigenstate(self, index, dtype=np.float32):
        return self.ct.eigenstate(index, dtype).ct

    def identity(self, dtype=np.float32):
        return self.ct.identity(dtype)

    def __hash__(self):
        return hash(('bra', self.ct))

    def __eq__(self, other):
        return isinstance(other, BraSpace) and self.ct == other.ct
