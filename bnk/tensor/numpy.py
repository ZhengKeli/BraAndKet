from typing import Iterable, Tuple, Union

import numpy as np

from .formal import FormalQTensor
from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class NumpyQTensor(FormalQTensor):

    # basic

    def __init__(self, spaces: Iterable[Space], values):
        spaces = tuple(spaces)
        values = np.asarray(values)

        for space, shape_n in zip(spaces, np.shape(values)):
            if shape_n == space.n:
                continue
            if isinstance(space, NumSpace) and shape_n == 1:
                continue
            raise ValueError(f"The spaces do not match the shape!")

        super().__init__(spaces)
        self._spaces = spaces
        self._values = values

    def _formal_getitem(self, *items: Tuple[Space, Union[int, slice, tuple]]):
        axis_list = []
        slice_list = []
        for spa, sli in items:
            axis = self._spaces.index(spa)
            axis_list.append(axis)
            slice_list.append(sli)

        for axis in np.arange(len(self._spaces)):
            if axis not in axis_list:
                axis_list.append(axis)

        values = np.transpose(self._values, axis_list)

        skip_slices = []
        for sli in slice_list:
            values = values[(*skip_slices, sli)]
            if not isinstance(sli, int):
                skip_slices.append(slice(None))

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

    def _formal_add(self, other):
        new_spaces = self._spaces
        new_values = self._values + other[new_spaces]
        return NumpyQTensor(new_spaces, new_values)

    def _formal_mul(self, other):
        new_spaces = self._spaces
        new_values = self._values * other
        return NumpyQTensor(new_spaces, new_values)

    # tensor operations

    @property
    def ct(self):
        new_spaces = tuple(space.ct for space in self._spaces if isinstance(space, HSpace))
        new_values = np.conjugate(self._values)
        return NumpyQTensor(new_spaces, new_values)

    def _formal_trace(self, *spaces: KetSpace):
        traced = self
        for ket_space in spaces:
            ket_axis = traced._spaces.index(ket_space)
            bra_axis = traced._spaces.index(ket_space.ct)
            new_space = tuple(space for axis, space in enumerate(traced.spaces) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced._values, axis1=ket_axis, axis2=bra_axis)
            traced = NumpyQTensor(new_space, new_values)
        return traced

    def _formal_matmul(self, other):
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
        new_other_spaces = [space for axis, space in enumerate(other_spaces) if axis not in other_dot_axes]

        new_spaces = [*new_self_spaces, *new_other_spaces]
        new_values = np.tensordot(self_values, other_values, (self_dot_axes, other_dot_axes))

        return NumpyQTensor(new_spaces, new_values)

    # space operations

    def _formal_broadcast(self, ket_spaces: Iterable[KetSpace], num_spaces: Iterable[NumSpace]):
        from ..math import prod
        broadcast_identity = prod(ket_space.identity() for ket_space in ket_spaces)
        new_tensor = self @ broadcast_identity

        num_spaces = tuple(num_spaces)
        if len(num_spaces) > 0:
            num_shape = [1] * len(num_spaces)
            new_shape = (*num_shape, *np.shape(new_tensor.values))
            new_values = np.reshape(new_tensor.values, new_shape)
            spaces = (*num_spaces, *new_tensor.spaces)
            new_tensor = NumpyQTensor(spaces, new_values)

        return new_tensor

    @staticmethod
    def inflate(flattened_values, flattened_spaces, *, copy=True):
        wrapped_spaces = [space for group in flattened_spaces for space in group]
        wrapped_shape = [space.n for space in wrapped_spaces]
        wrapped_values = np.reshape(flattened_values, wrapped_shape)

        if copy:
            wrapped_values = np.copy(wrapped_values)

        tensor = NumpyQTensor(wrapped_spaces, wrapped_values)

        return tensor
