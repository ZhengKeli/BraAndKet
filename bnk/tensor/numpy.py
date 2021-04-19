from typing import Iterable

import numpy as np

from .abstract import QTensor
from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class NumpyQTensor(QTensor):
    def __init__(self, spaces: Iterable[Space], values):
        spaces = tuple(spaces)
        values = np.asarray(values)

        if len(set(spaces)) != len(spaces):
            raise ValueError("There are duplicated spaces!")

        for space, shape_n in zip(spaces, np.shape(values)):
            if shape_n == space.n:
                continue
            if isinstance(space, NumSpace) and shape_n == 1:
                continue
            raise ValueError(f"The spaces do not match the shape!")

        self._spaces = spaces
        self._values = values

    @property
    def spaces(self):
        return self._spaces

    @property
    def values(self):
        return self._values

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __repr__(self):
        return f"QTensor(spaces={self.spaces}, values={self.values})"

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

    # space operations

    def transposed(self, new_spaces: Iterable[Space]):
        if self.spaces == new_spaces:
            return self
        new_axes = [self.spaces.index(new_space) for new_space in new_spaces]
        new_values = np.transpose(self.values, axes=new_axes)
        return NumpyQTensor(new_spaces, new_values)

    def broadcast(self, broadcast_spaces: Iterable[Space]):
        if len(self.spaces) == 0:
            if self.values == 0:
                values = np.zeros([space.n for space in broadcast_spaces])
                return NumpyQTensor(broadcast_spaces, values)

        broadcast_num_spaces = []
        broadcast_identity = one

        broadcast_spaces = set(broadcast_spaces)
        while broadcast_spaces:
            broadcast_space = broadcast_spaces.pop()

            if broadcast_space in self.spaces:
                continue

            if isinstance(broadcast_space, NumSpace):
                broadcast_num_spaces.append(broadcast_space)
                continue

            if isinstance(broadcast_space, HSpace):
                if broadcast_space.ct not in broadcast_spaces:
                    raise TypeError(f"Can not broadcast unpaired space {broadcast_space}.")
                broadcast_spaces.remove(broadcast_space.ct)
                broadcast_identity @= broadcast_space.ket.identity(self.dtype)
                continue

            raise TypeError(f"Unsupported custom type {type(broadcast_space)}!")

        new_tensor = self @ broadcast_identity

        if broadcast_num_spaces:
            num_shape = [1] * len(broadcast_num_spaces)
            new_shape = (*num_shape, *np.shape(new_tensor.values))
            new_values = np.reshape(new_tensor.values, new_shape)
            broadcast_spaces = (*broadcast_num_spaces, *new_tensor.spaces)
            new_tensor = NumpyQTensor(broadcast_spaces, new_values)

        return new_tensor

    # tensor operations

    @property
    def ct(self):
        new_spaces = (space.ct for space in self.spaces if isinstance(space, HSpace))
        new_values = np.conjugate(self.values)
        return NumpyQTensor(new_spaces, new_values)

    def trace(self, *spaces: HSpace):
        if len(spaces) == 0:
            spaces = self.spaces

        traced = self
        for space in spaces:
            if not isinstance(space, KetSpace):
                continue
            ket_space = space
            ket_axis = traced.spaces.index(ket_space)
            bra_axis = traced.spaces.index(ket_space.ct)
            new_space = tuple(space for axis, space in enumerate(traced.spaces) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced.values, axis1=ket_axis, axis2=bra_axis)
            traced = NumpyQTensor(new_space, new_values)

        return traced

    def __matmul__(self, other):
        if not isinstance(other, QTensor):
            other_values = np.asarray(other)
            other_shape = np.shape(other_values)
            if not other_shape:
                return self * other
            other_space = [NumSpace(shape_n) for shape_n in other_shape]
            other_tensor = NumpyQTensor(other_space, other_values)
            return self @ other_tensor

        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_space in enumerate(self.spaces):
            # KetSpace ignored
            if isinstance(self_space, BraSpace):
                try:
                    other_axis = other.spaces.index(self_space.ct)
                    self_dot_axes.append(self_axis)
                    other_dot_axes.append(other_axis)
                except ValueError:
                    pass
            elif isinstance(self_space, NumSpace):
                try:
                    other.spaces.index(self_space)
                    raise NotImplementedError(f"Found {self_space}. Not implemented matmul for NumSpace!")
                except ValueError:
                    pass

        new_self_spaces = [space for axis, space in enumerate(self.spaces) if axis not in self_dot_axes]
        new_other_spaces = [space for axis, space in enumerate(other.spaces) if axis not in other_dot_axes]

        new_spaces = [*new_self_spaces, *new_other_spaces]
        new_values = np.tensordot(self.values, other.values, (self_dot_axes, other_dot_axes))

        return NumpyQTensor(new_spaces, new_values)

    def __rmatmul__(self, other):
        return other * self

    # linear operations

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

        broadcast_self = self.broadcast(other.spaces)
        broadcast_other = other.broadcast(self.spaces)

        new_spaces = broadcast_self.spaces
        broadcast_other = broadcast_other.transposed(new_spaces)

        new_values = broadcast_self.values + broadcast_other.values
        return NumpyQTensor(new_spaces, new_values)

    def __sub__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

        broadcast_self = self.broadcast(other.spaces)
        broadcast_other = other.broadcast(self.spaces)

        new_spaces = broadcast_self.spaces
        broadcast_other = broadcast_other.transposed(new_spaces)

        new_values = broadcast_self.values - broadcast_other.values
        return NumpyQTensor(new_spaces, new_values)

    def __mul__(self, other):
        if isinstance(other, QTensor):
            raise TypeError("Please use matmul operator \"@\" for QTensors.")
        if len(np.shape(other)) > 0:
            raise TypeError("QTensor can perform * and / only with a scalar.")
        new_spaces = self.spaces
        new_values = self.values * other
        return NumpyQTensor(new_spaces, new_values)

    def __truediv__(self, other):
        if isinstance(other, QTensor):
            raise TypeError("Please use matmul operator \"@\" for QTensors.")
        if len(np.shape(other)) > 0:
            raise TypeError("QTensor can perform * and / only with a scalar.")
        new_spaces = self.spaces
        new_values = self.values / other
        return NumpyQTensor(new_spaces, new_values)


zero = NumpyQTensor([], np.zeros([], np.float))

one = NumpyQTensor([], np.ones([], np.float))
