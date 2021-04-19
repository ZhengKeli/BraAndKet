import abc
from typing import Tuple, Iterable

import numpy as np

from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class QTensor(abc.ABC):

    # basic

    @property
    @abc.abstractmethod
    def spaces(self) -> Tuple[Space]:
        pass

    @property
    def shape(self):
        return tuple(space.n for space in self.spaces)

    @property
    @abc.abstractmethod
    def values(self):
        """ multi-dimensional values
        :return: multi-dimensional values
        """
        pass

    def __eq__(self, other):
        if self is other:
            return True
        if other == 0:
            return len(self.spaces) == 0 and self.values == 0
        if not isinstance(other, QTensor):
            return False

        broadcast_self = self.broadcast(other.spaces)
        broadcast_other = other.broadcast(self.spaces)
        broadcast_other = broadcast_other.transposed(broadcast_self.spaces)
        return np.all(broadcast_self.values == broadcast_other.values)

    # space operations

    @abc.abstractmethod
    def transposed(self, new_spaces: Iterable[Space]):
        pass

    @abc.abstractmethod
    def broadcast(self, broadcast_spaces: Iterable[Space]):
        pass

    # flatten & wrap

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

        transposed = self.transposed([*num_spaces, *ket_spaces, *bra_spaces])
        flattened_values = np.reshape(transposed.values, flattened_shape)

        return flattened_spaces, flattened_values

    @staticmethod
    def wrap(flattened_values, flattened_spaces, spaces=None, copy=True):
        from .numpy import NumpyQTensor

        wrapped_spaces = [space for group in flattened_spaces for space in group]
        wrapped_shape = [space.n for space in wrapped_spaces]
        wrapped_values = np.reshape(flattened_values, wrapped_shape)

        if copy:
            wrapped_values = np.copy(wrapped_values)

        tensor = NumpyQTensor(wrapped_spaces, wrapped_values)
        if spaces:
            tensor = tensor.transposed(spaces)

        return tensor

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
            psi /= float(psi.ct @ psi)

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
            rho /= float(rho.trace())

        return rho

    # tensor operations

    @property
    @abc.abstractmethod
    def ct(self):
        pass

    @abc.abstractmethod
    def trace(self, *spaces: HSpace):
        pass

    @abc.abstractmethod
    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        return other * self

    # linear operations

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self

    @abc.abstractmethod
    def __add__(self, other):
        pass

    def __radd__(self, other):
        if other == 0:
            return self
        raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if other == 0:
            return -self
        raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1.0 / other)

    # single item operations

    def item(self):
        if len(self.shape) > 0:
            raise ValueError("Can not convert Tensor with rank>0 to float!")
        return self.values

    def __float__(self):
        return float(self.item())
