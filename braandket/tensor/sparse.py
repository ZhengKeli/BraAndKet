from typing import Tuple, Union, Iterable

import numpy as np

from .formal import FormalQTensor
from ..space import Space, NumSpace, HSpace, BraSpace, KetSpace


class SparseQTensor(FormalQTensor):

    # basic

    def __init__(self, spaces: Iterable[Space], values):
        super().__init__(spaces)

        _values = {}
        for coordinate, value in values:
            coordinate = tuple(int(i) for i in coordinate)
            if len(coordinate) != len(self.spaces):
                raise ValueError("The shape of coordinates does not match the spaces")
            for axis, i in enumerate(coordinate):
                if not 0 <= i <= self.spaces[axis].n:
                    raise ValueError("The coordinates is out of bounds!")

            existed = _values.get(coordinate)
            if existed is None:
                _values[coordinate] = value
            else:
                new_value = existed + value
                if new_value == 0:
                    del _values[coordinate]
                else:
                    _values[coordinate] = new_value

        self._values = _values

    @property
    def values(self):
        return self._values.items()

    def _formal_get(self, *items: Tuple[Space, Union[int, slice, tuple]], dtype):
        items = tuple(items)
        spaces = tuple(spa for spa, _ in items)
        axes = tuple(self.spaces.index(space) for space in spaces)

        new_coordinates = []
        new_values = []
        for coordinate, value in self.values:
            coordinate = tuple(coordinate[axis] for axis in axes)

            new_coordinate = [[]]
            for i, (spa, sli) in zip(coordinate, items):
                if isinstance(sli, slice):
                    if sli == slice(None):
                        new_i = i
                        for coo in new_coordinate:
                            coo.append(new_i)
                        continue
                    else:
                        sli = tuple(range(spa.n))[sli]
                if isinstance(sli, tuple):
                    new_i_list = []
                    for new_i, sli_i in enumerate(sli):
                        if sli_i == i:
                            new_i_list.append(new_i)
                    if len(new_i_list) == 0:
                        new_coordinate = None
                        break
                    elif len(new_i_list) == 1:
                        new_i = new_i_list[0]
                        for coo in new_coordinate:
                            coo.append(new_i)
                    else:
                        new_coordinate = [
                            [*coo, new_i]
                            for new_i in new_i_list
                            for coo in new_coordinate
                        ]
                elif isinstance(sli, int):
                    if i != sli:
                        new_coordinate = None
                        break
                else:
                    new_coordinate = None
                    break

            if new_coordinate is None:
                continue

            new_coordinates.extend(new_coordinate)
            new_values.extend([value] * len(new_coordinate))

        new_shape = []
        for spa, sli in items:
            if isinstance(sli, slice):
                if sli == slice(None):
                    n = spa.n
                else:
                    n = len(tuple(range(spa.n))[sli])
                new_shape.append(n)
            elif isinstance(sli, tuple):
                n = len(sli)
                new_shape.append(n)
            elif isinstance(sli, int):
                pass
            else:
                assert False

        if len(new_shape) == 0:
            return np.sum(new_values, dtype=dtype)

        new_values = np.asarray(new_values, dtype=dtype)
        new_coordinates = tuple(tuple(new_coordinate) for new_coordinate in new_coordinates)
        new_array = np.zeros(new_shape, dtype=new_values.dtype)

        for new_coordinate, new_value in zip(new_coordinates, new_values):
            new_array[new_coordinate] += new_value
        return new_array

    def to_dense(self):
        spaces = self.spaces
        values = self[spaces]
        from .numpy import NumpyQTensor
        return NumpyQTensor(spaces, values)

    def copy(self):
        return SparseQTensor(self.spaces, self.values)

    def __repr__(self):
        spaces = self.spaces
        values = self._values
        return f"{self.__class__.__name__}(spaces={repr(spaces)}, values={repr(values)})"

    # scalar operations

    @classmethod
    def from_scalar(cls, scalar):
        return SparseQTensor(tuple(), ((tuple(), scalar),))

    # linear operations

    def _formal_add(self, other):
        if not isinstance(other, SparseQTensor):
            return self.to_dense() + other

        new_spaces = self.spaces
        self_axes = tuple(other.spaces.index(space) for space in new_spaces)

        def iter_new_values():
            for coordinate, value in self.values:
                yield coordinate, value
            for other_coordinate, other_value in other.values:
                coordinate = tuple(other_coordinate[axis] for axis in self_axes)
                value = other_value
                yield coordinate, value

        new_values = iter_new_values()
        return SparseQTensor(new_spaces, new_values)

    def _formal_mul(self, other):
        new_spaces = self.spaces
        new_values = ((coordinate, value * other) for coordinate, value in self.values)
        return SparseQTensor(new_spaces, new_values)

    # tensor operations

    @property
    def ct(self):
        new_spaces = tuple(space.ct for space in self.spaces if isinstance(space, HSpace))
        new_values = ((coordinate, np.conjugate(value)) for coordinate, value in self.values)
        return SparseQTensor(new_spaces, new_values)

    def _formal_trace(self, *spaces: KetSpace):
        axes_ket = tuple(self.spaces.index(space) for space in spaces)
        axes_bra = tuple(self.spaces.index(space.ct) for space in spaces)
        axes_others = tuple(i for i in range(len(self.spaces)) if i not in axes_ket and i not in axes_bra)

        def iter_new_values():
            for coordinate, value in self.values:
                if all(coordinate[axis_ket] == coordinate[axis_bra] for axis_ket, axis_bra in zip(axes_ket, axes_bra)):
                    new_coordinate = tuple(coordinate[axis] for axis in axes_others)
                    new_value = value
                    yield new_coordinate, new_value

        new_spaces = tuple(self.spaces[axis] for axis in axes_others)
        new_values = iter_new_values()
        return SparseQTensor(new_spaces, new_values)

    def _formal_matmul(self, other):
        from ..tensor import NumpyQTensor
        if not isinstance(other, (NumpyQTensor, SparseQTensor)):
            raise TypeError(f"Unsupported type {type(other)}")

        self_spaces = tuple(self.spaces)
        other_spaces = tuple(other.spaces)

        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_space in enumerate(self_spaces):
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

        new_self_axes = tuple(axis for axis in range(len(self_spaces)) if axis not in self_dot_axes)
        new_other_axes = tuple(axis for axis in range(len(other_spaces)) if axis not in other_dot_axes)
        new_spaces = tuple(self_spaces[axis] for axis in new_self_axes) + \
                     tuple(other_spaces[axis] for axis in new_other_axes)

        if isinstance(other, SparseQTensor):
            def iter_new_values():
                for self_coordinate, self_value in self.values:
                    for other_coordinate, other_value in other.values:
                        if all(self_coordinate[axis_bra] == other_coordinate[axis_ket]
                               for axis_bra, axis_ket in zip(self_dot_axes, other_dot_axes)):
                            new_coordinate = tuple(self_coordinate[axis] for axis in new_self_axes) + \
                                             tuple(other_coordinate[axis] for axis in new_other_axes)
                            new_value = self_value * other_value
                            yield new_coordinate, new_value

            new_values = iter_new_values()
        elif isinstance(other, NumpyQTensor):
            other_values = other.values

            def iter_other_values(self_coordinate):
                other_new_coordinates = np.mgrid[tuple(slice(other_spaces[axis].n) for axis in new_other_axes)]
                other_new_coordinates = np.reshape(np.stack(other_new_coordinates, axis=-1), [-1, len(new_other_axes)])
                other_coordinates = np.zeros([len(other_new_coordinates), len(other_spaces)])
                other_coordinates[new_other_axes, :] = other_new_coordinates
                other_coordinates[other_dot_axes, :] = np.expand_dims(self_coordinate, -1)
                for other_coordinate in other_coordinates:
                    yield other_coordinate, other_values[tuple(other_coordinate)]

            def iter_new_values():
                for self_coordinate, self_value in self.values:
                    for other_coordinate, other_value in iter_other_values(self_coordinate):
                        new_coordinate = tuple(self_coordinate[axis] for axis in new_self_axes) + \
                                         tuple(other_coordinate[axis] for axis in new_other_axes)
                        new_value = self_value * other_value
                        yield new_coordinate, new_value

            new_values = iter_new_values()
        else:
            assert False
        return SparseQTensor(new_spaces, new_values)

    # space operations

    def _formal_broadcast(self, ket_spaces: Iterable[KetSpace], num_spaces: Iterable[NumSpace]):
        from ..utils import prod
        broadcast_identity = prod(ket_space.identity() for ket_space in ket_spaces)
        new_tensor = self @ broadcast_identity

        num_spaces = tuple(num_spaces)
        if len(num_spaces) > 0:
            raise TypeError("NumSpace is not supported yet!")

        return new_tensor

    def _formal_flatten(self, ket_spaces, bra_spaces, *, dtype, sparse):
        ket_axes = tuple(self.spaces.index(space) for space in ket_spaces)
        bra_axes = tuple(self.spaces.index(space) for space in bra_spaces)

        ket_shape = tuple(space.n for space in ket_spaces)
        bra_shape = tuple(space.n for space in bra_spaces)
        new_shape = (np.prod(ket_shape, dtype=int), np.prod(bra_shape, dtype=int))

        new_ket_indices = []
        new_bra_indices = []
        new_values = []
        for coordinate, value in self.values:
            ket_coordinate = tuple(coordinate[axis] for axis in ket_axes)
            bra_coordinate = tuple(coordinate[axis] for axis in bra_axes)

            if len(ket_coordinate) > 0:
                ket_index = np.ravel_multi_index(ket_coordinate, ket_shape)
            else:
                ket_index = 0

            if len(bra_coordinate) > 0:
                bra_index = np.ravel_multi_index(bra_coordinate, bra_shape)
            else:
                bra_index = 0

            new_ket_indices.append(ket_index)
            new_bra_indices.append(bra_index)
            new_values.append(value)

        if sparse:
            from scipy.sparse import coo_matrix
            new_matrix = coo_matrix((new_values, (new_ket_indices, new_bra_indices)), new_shape, dtype=dtype)
            return new_matrix
        else:
            new_array = self.get(*ket_spaces, *bra_spaces, dtype=dtype)
            new_array = np.reshape(new_array, new_shape)
            return new_array

    @classmethod
    def _formal_inflate(cls, flattened, ket_spaces, bra_spaces, *, copy=True):
        ket_spaces = tuple(ket_spaces)
        bra_spaces = tuple(bra_spaces)
        ket_shape = tuple(space.n for space in ket_spaces)
        bra_shape = tuple(space.n for space in bra_spaces)
        try:
            from scipy.sparse import coo_matrix
            flattened = coo_matrix(flattened)

            if len(ket_shape) > 0:
                ket_coordinates = np.transpose(np.unravel_index(flattened.row, ket_shape))
            else:
                ket_coordinates = tuple(tuple()) * len(flattened.row)

            if len(bra_shape) > 0:
                bra_coordinates = np.transpose(np.unravel_index(flattened.col, bra_shape))
            else:
                bra_coordinates = tuple(tuple()) * len(flattened.col)

            values = (
                ((*ket_coordinate, *bra_coordinate), value)
                for ket_coordinate, bra_coordinate, value in zip(ket_coordinates, bra_coordinates, flattened.data)
            )
            return SparseQTensor((*ket_spaces, *bra_spaces), values)
        except ImportError:
            rows, cols = np.nonzero(flattened)
            ket_coordinates = np.transpose(np.unravel_index(rows, ket_shape))
            bra_coordinates = np.transpose(np.unravel_index(cols, bra_shape))
            values = (
                ((*ket_coordinate, *bra_coordinate), flattened[row, col])
                for row, col, ket_coordinate, bra_coordinate in zip(rows, cols, ket_coordinates, bra_coordinates)
            )
            return SparseQTensor((*ket_spaces, *bra_spaces), values)
