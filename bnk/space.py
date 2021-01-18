import abc
import warnings
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

    def __repr__(self):
        if self.name is None:
            return f"{NumSpace.__name__}({self.n})"
        else:
            return f"{NumSpace.__name__}({self.n}, name={self.name})"


class HSpace(Space, abc.ABC):
    @property
    @abc.abstractmethod
    def ct(self):
        """
        :rtype: HSpace
        """
        pass

    @property
    @abc.abstractmethod
    def ket(self):
        """
        :rtype: KetSpace
        """
        pass

    @property
    @abc.abstractmethod
    def bra(self):
        """
        :rtype: BraSpace
        """
        pass


class KetSpace(HSpace):
    def __init__(self, n, name=None):
        super().__init__()
        self._n = n
        self._name = name

        self._eigenstates = np.eye(self.n, dtype=np.bool)
        self._bra = BraSpace(self)

    @property
    def n(self):
        return self._n

    @property
    def name(self):
        return self._name

    def __repr__(self):
        if self.name is None:
            return f"{KetSpace.__name__}({self.n})"
        else:
            return f"{KetSpace.__name__}({self.n}, name={self.name})"

    @property
    def ct(self):
        return self._bra

    @property
    def ket(self):
        return self

    @property
    def bra(self):
        return self._bra

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

    def operator(self, ket_index, bra_index, dtype=np.float32):
        return self.eigenstate(ket_index, dtype) @ self.eigenstate(bra_index, dtype).ct

    def projector(self, index, dtype=np.float32):
        return self.operator(index, index, dtype)


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

    @property
    def n(self):
        return self._ket.n

    @property
    def name(self):
        return self._ket.name

    def __repr__(self):
        if self.name is None:
            return f"{BraSpace.__name__}({self.n})"
        else:
            return f"{BraSpace.__name__}({self.n}, name={self.name})"

    @property
    def ct(self):
        return self._ket

    @property
    def ket(self):
        return self._ket

    @property
    def bra(self):
        return self
