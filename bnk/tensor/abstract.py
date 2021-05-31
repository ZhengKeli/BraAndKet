import abc
from typing import Tuple, Iterable, Set, Union

import numpy as np

from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class QTensor(abc.ABC):

    # basic

    @property
    @abc.abstractmethod
    def spaces(self) -> Set[Space]:
        """ the spaces/dimensions of this tensor

        QTensor use spaces as their dimensions.
        The spaces are not required to be ordered,
        but they are not allowed to be duplicated.

        :return: a set of Spaces.
        """

    @abc.abstractmethod
    def __getitem__(self, items):
        """ get specific values from this tensor """

    def __iter__(self):
        """ iteration is NOT allowed """
        raise TypeError("QTensor is not allowed to be iterated.")

    def __eq__(self, other):
        if self is other:
            return True
        if self.is_scalar:
            return self.scalar() == other
        if not isinstance(other, QTensor):
            return False

        union_spaces = tuple(self.spaces.union(other.spaces))
        self_broadcast = self.broadcast(union_spaces)
        other_broadcast = other.broadcast(union_spaces)

        self_values = np.asarray(self_broadcast[union_spaces])
        other_values = np.asarray(other_broadcast[union_spaces])
        return np.all(self_values == other_values)

    @abc.abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        return self.__copy__()

    def __repr__(self):
        spaces = tuple(self.spaces)
        values = self[spaces]
        return f"{self.__class__.__name__}(spaces={repr(spaces)}, values={repr(values)})"

    # scalar operations

    @property
    def is_scalar(self):
        return len(self.spaces) == 0

    def scalar(self):
        """ get a scalar value from zero-dimension tensor """
        if not self.is_scalar:
            raise ValueError(f"This QTensor is not a scalar. It has {len(self.spaces)} spaces.")
        return self[tuple()]

    def __float__(self):
        return float(self.scalar())

    # linear operations

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1) * self

    @abc.abstractmethod
    def __add__(self, other):
        """
        :rtype: QTensor
        """
        pass

    def __radd__(self, other):
        """
        :param other: a scalar
        :rtype: QTensor
        """
        return self + other

    def __sub__(self, other):
        """
        :rtype: QTensor
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        :param other: a scalar
        :rtype: QTensor
        """
        return -self + other

    @abc.abstractmethod
    def __mul__(self, other):
        """
        :param other: a scalar or zero-dimensional QTensor
        :rtype: QTensor
        """
        pass

    def __rmul__(self, other):
        """
        :param other: a scalar
        :rtype: QTensor
        """
        return self * other

    def __truediv__(self, other):
        """
        :param other: a scalar or zero-dimensional QTensor
        :rtype: QTensor
        """
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        if self.is_scalar:
            return other / self.scalar()
        raise TypeError("Can not divide an QTensor (except a zero-dimensional one)")

    # tensor operations

    @property
    @abc.abstractmethod
    def ct(self):
        """ conjugate transpose

        :rtype: QTensor
        """
        pass

    @abc.abstractmethod
    def trace(self, *spaces: KetSpace):
        """ trace by spaces

        :rtype: QTensor
        """
        pass

    @abc.abstractmethod
    def __matmul__(self, other):
        """ tensor product

        Normal multiplication is performed when other is a scalar.

        :param other: a QTensor or scalar
        :rtype: QTensor
        """
        pass

    def __rmatmul__(self, other):
        """

        :param other: a scalar
        :return: normal product
        """
        return other * self

    # space operations

    @abc.abstractmethod
    def broadcast(self, broadcast_spaces: Iterable[Space]):
        pass

    def flatten(self):
        num_spaces = []
        ket_spaces = []
        bra_spaces = []
        for space in self.spaces:
            if isinstance(space, KetSpace):
                ket_spaces.append(space)
            elif isinstance(space, BraSpace):
                bra_spaces.append(space)
            else:
                num_spaces.append(space)

        num_spaces = tuple(sorted(num_spaces, key=lambda sp: (-sp.n, id(sp))))
        ket_spaces = tuple(sorted(ket_spaces, key=lambda sp: (-sp.n, id(sp))))
        bra_spaces = tuple(sorted(bra_spaces, key=lambda sp: (-sp.n, id(sp.ket))))

        flattened_num_space = np.prod([space.n for space in num_spaces], dtype=int)
        flattened_ket_space = np.prod([space.n for space in ket_spaces], dtype=int)
        flattened_bra_space = np.prod([space.n for space in bra_spaces], dtype=int)

        if flattened_num_space == 1:
            flattened_spaces = ket_spaces, bra_spaces
            flattened_shape = [flattened_ket_space, flattened_bra_space]
        else:
            flattened_spaces = num_spaces, ket_spaces, bra_spaces
            flattened_shape = [flattened_num_space, flattened_ket_space, flattened_bra_space]

        flattened_values = self[(*num_spaces, *ket_spaces, *bra_spaces)]
        flattened_values = np.reshape(flattened_values, flattened_shape)

        return flattened_spaces, flattened_values

    @staticmethod
    def inflate(flattened_values, flattened_spaces, *, copy=True):
        from .numpy import NumpyQTensor
        return NumpyQTensor.inflate(flattened_values, flattened_spaces)

    @property
    def flattened_values(self):
        _, flattened_values = self.flatten()
        return flattened_values

    # psi & rho

    @property
    def is_psi(self):
        for space in self.spaces:
            if isinstance(space, NumSpace):
                continue
            elif isinstance(space, KetSpace):
                continue
            else:
                return False
        return True

    def as_psi(self, normalize=True):
        if not self.is_psi:
            raise TypeError("This tensor is not a ket vector!")
        psi = self

        if normalize:
            psi /= (psi.ct @ psi)

        return psi

    @property
    def is_rho(self):
        spaces = set(self.spaces)
        while spaces:
            space = spaces.pop()
            if isinstance(space, NumSpace):
                continue
            if isinstance(space, HSpace):
                if space.ct not in spaces:
                    return False
                spaces.remove(space.ct)
        return True

    def as_rho(self, normalize=True):
        if not self.is_rho:
            raise ValueError("This tensor is not a density matrix!")
        rho = self

        if normalize:
            rho /= rho.trace()

        return rho
