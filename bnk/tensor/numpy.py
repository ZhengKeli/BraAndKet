from typing import Iterable, Tuple, Union, Set

import numpy as np

from .abstract import QTensor
from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class NumpyQTensor(QTensor):

    # basic

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
    def spaces(self) -> Set[Space]:
        return set(self._spaces)

    def __formal_getitem__(self, *items: Tuple[Space, Union[int, slice]]):
        axis_list = []
        slice_list = []
        for spa, sli in items:
            axis = self._spaces.index(spa)
            axis_list.append(axis)
            slice_list.append(sli)

        for axis in np.arange(len(self._spaces)):
            if axis not in axis_list:
                axis_list.append(axis)
                slice_list.append(slice(None))

        values = np.transpose(self._values, axis_list)
        values = values[tuple(slice_list)]
        return values

    def __copy__(self):
        return NumpyQTensor(self._spaces, self._values.copy())

    def __repr__(self):
        return f"NumpyQTensor(spaces={repr(self._spaces)}, values={self._values})"

    # scalar operations

    @staticmethod
    def from_scalar(scalar):
        return NumpyQTensor([], scalar)

    # linear operations

    def __add__(self, other):
        if other == 0:
            return self
        if self.is_scalar:
            return NumpyQTensor.from_scalar(self.scalar() + other)
        if not isinstance(other, QTensor):
            raise TypeError(f"Can not perform + operation with {other}: this QTensor is not zero-dimensional.")

        new_spaces = tuple(self.spaces.union(other.spaces))
        self_broadcast = self.broadcast(new_spaces)
        other_broadcast = other.broadcast(new_spaces)

        new_spaces = self_broadcast.spaces
        new_values = self_broadcast[new_spaces] + other_broadcast[new_spaces]
        return NumpyQTensor(new_spaces, new_values)

    def __mul__(self, other):
        if other == 1:
            return self
        if self.is_scalar:
            return NumpyQTensor.from_scalar(self.scalar() * other)
        if isinstance(other, QTensor):
            if not other.is_scalar:
                raise TypeError("Please use matmul operator \"@\" for tensor product.")
            other = other.scalar()
        else:
            other = np.asarray(other).item()

        new_spaces = self._spaces
        new_values = self._values * other
        return NumpyQTensor(new_spaces, new_values)

    # tensor operations

    @property
    def ct(self):
        new_spaces = tuple(space.ct for space in self._spaces if isinstance(space, HSpace))
        new_values = np.conjugate(self._values)
        return NumpyQTensor(new_spaces, new_values)

    def trace(self, *spaces: KetSpace):
        if len(spaces) == 0:
            spaces = self._spaces

        traced = self
        for ket_space in spaces:
            ket_axis = traced._spaces.index(ket_space)
            bra_axis = traced._spaces.index(ket_space.ct)
            new_space = tuple(space for axis, space in enumerate(traced.spaces) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced._values, axis1=ket_axis, axis2=bra_axis)
            traced = NumpyQTensor(new_space, new_values)

        return traced

    def __matmul__(self, other):
        if not isinstance(other, QTensor):
            return self * other

        self_spaces = self._spaces
        self_values = self._values

        other_spaces = tuple(other.spaces)
        other_values = other[other_spaces]

        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_space in enumerate(self.spaces):
            # KetSpace ignored
            if isinstance(self_space, BraSpace):
                try:
                    other_axis = other_spaces.index(self_space.ct)
                    self_dot_axes.append(self_axis)
                    other_dot_axes.append(other_axis)
                except ValueError:
                    pass
            elif isinstance(self_space, NumSpace):
                try:
                    other_spaces.index(self_space)
                    raise NotImplementedError(f"Found {self_space}. Not implemented matmul for NumSpace!")
                except ValueError:
                    pass

        new_self_spaces = [space for axis, space in enumerate(self_spaces) if axis not in self_dot_axes]
        new_other_spaces = [space for axis, space in enumerate(self_values) if axis not in other_dot_axes]

        new_spaces = [*new_self_spaces, *new_other_spaces]
        new_values = np.tensordot(self_values, other_values, (self_dot_axes, other_dot_axes))

        return NumpyQTensor(new_spaces, new_values)

    # space operations

    @staticmethod
    def inflate(flattened_values, flattened_spaces, *, copy=True):
        wrapped_spaces = [space for group in flattened_spaces for space in group]
        wrapped_shape = [space.n for space in wrapped_spaces]
        wrapped_values = np.reshape(flattened_values, wrapped_shape)

        if copy:
            wrapped_values = np.copy(wrapped_values)

        tensor = NumpyQTensor(wrapped_spaces, wrapped_values)

        return tensor


zero = NumpyQTensor.from_scalar(0.0)

one = NumpyQTensor.from_scalar(1.0)
