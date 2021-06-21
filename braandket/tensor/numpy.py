from typing import Iterable, Tuple, Union

import numpy as np

from .formal import FormalQTensor
from ..space import Space, HSpace, NumSpace, BraSpace, KetSpace


class NumpyQTensor(FormalQTensor):

    # basic

    def __init__(self, spaces: Iterable[Space], values):
        super().__init__(spaces)
        values = np.asarray(values)

        for space, shape_n in zip(spaces, np.shape(values)):
            if shape_n == space.n:
                continue
            if isinstance(space, NumSpace) and shape_n == 1:
                continue
            raise ValueError(f"The spaces do not match the shape!")

        self._values = values

    @property
    def values(self):
        return self._values

    def _formal_get(self, *items: Tuple[Space, Union[int, slice, tuple]], dtype):
        axis_list = []
        slice_list = []
        for spa, sli in items:
            axis = self.spaces.index(spa)
            axis_list.append(axis)
            slice_list.append(sli)

        for axis in np.arange(len(self.spaces)):
            if axis not in axis_list:
                axis_list.append(axis)

        values = np.transpose(self.values, axis_list)

        skip_slices = []
        for sli in slice_list:
            values = values[(*skip_slices, sli)]
            if not isinstance(sli, int):
                skip_slices.append(slice(None))

        return values

    def copy(self):
        return NumpyQTensor(self.spaces, self.values.copy())

    def __repr__(self):
        return f"NumpyQTensor(spaces={repr(self.spaces)}, values={self.values})"

    # scalar operations

    @classmethod
    def from_scalar(cls, scalar):
        return NumpyQTensor([], scalar)

    # linear operations

    def _formal_add(self, other):
        new_spaces = self.spaces
        new_values = self.values + other[new_spaces]
        return NumpyQTensor(new_spaces, new_values)

    def _formal_mul(self, other):
        new_spaces = self.spaces
        new_values = self.values * other
        return NumpyQTensor(new_spaces, new_values)

    # tensor operations

    @property
    def ct(self):
        new_spaces = tuple(space.ct for space in self.spaces if isinstance(space, HSpace))
        new_values = np.conjugate(self.values)
        return NumpyQTensor(new_spaces, new_values)

    def _formal_trace(self, *spaces: KetSpace):
        traced = self
        for ket_space in spaces:
            ket_axis = traced.spaces.index(ket_space)
            bra_axis = traced.spaces.index(ket_space.ct)
            new_space = tuple(space for axis, space in enumerate(traced.spaces) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced.values, axis1=ket_axis, axis2=bra_axis)
            traced = NumpyQTensor(new_space, new_values)
        return traced

    def _formal_matmul(self, other):
        self_spaces = self.spaces
        self_values = self.values

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
        from ..utils import prod
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

    def _formal_flatten(self, ket_spaces, bra_spaces, *, dtype):
        ket_shape = [space.n for space in ket_spaces]
        bra_shape = [space.n for space in bra_spaces]
        flattened_shape = (np.prod(ket_shape, dtype=int), np.prod(bra_shape, dtype=int))

        flattened_values = self.get(*ket_spaces, *bra_spaces, dtype=dtype)
        flattened_values = np.reshape(flattened_values, flattened_shape)
        return flattened_values

    @classmethod
    def _formal_inflate(cls, flattened, ket_spaces, bra_spaces, *, copy=True):
        wrapped_spaces = [*ket_spaces, *bra_spaces]
        wrapped_shape = [space.n for space in wrapped_spaces]
        wrapped_values = np.reshape(flattened, wrapped_shape)
        return NumpyQTensor(wrapped_spaces, wrapped_values)
