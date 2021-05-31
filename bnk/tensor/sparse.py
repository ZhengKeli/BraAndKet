from typing import Tuple, Union, Iterable, Any

import numpy as np

from .formal import FormalQTensor
from ..space import Space, KetSpace, HSpace, BraSpace, NumSpace


class SparseQTensor(FormalQTensor):

    # basic

    def __init__(self, spaces: Iterable[Space], values: Union[dict, Iterable[Tuple[Tuple[int, ...], Any]]]):
        spaces = tuple(spaces)
        super().__init__(spaces)

        if isinstance(values, dict):
            values = values.items()

        values_map = {}
        for coordinate, value in values:
            coordinate = tuple(int(i) for i in coordinate)
            if len(coordinate) != len(spaces):
                raise ValueError("The shape of coordinates does not match the spaces")
            for axis, i in enumerate(coordinate):
                if not 0 <= i <= spaces[axis].n:
                    raise ValueError("The coordinates is out of bounds!")

            existed = values_map.get(coordinate)
            if existed is None:
                values_map[coordinate] = value
            else:
                new_value = existed + value
                if new_value == 0:
                    del values_map[coordinate]
                else:
                    values_map[coordinate] = new_value

        self._spaces = spaces
        self._values = values_map

    def _formal_getitem(self, *items: Tuple[Space, Union[int, slice, tuple]]):
        items = tuple(items)
        spaces = tuple(spa for spa, _ in items)
        axes = tuple(self._spaces.index(space) for space in spaces)

        new_coordinates = []
        new_values = []
        for coordinate, value in self._values.items():
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
            return new_values[0]

        new_indices = np.ravel_multi_index(np.transpose(new_coordinates), new_shape)
        new_values = np.asarray(new_values)
        new_array = np.zeros(new_shape, dtype=new_values.dtype)
        new_array.put(new_indices, new_values)
        return new_array

    def to_dense(self):
        spaces = self._spaces
        shape = tuple(space.n for space in spaces)
        coordinates, values = zip(*self._values.items())
        indices = np.ravel_multi_index(np.transpose(np.asarray(coordinates)))
        values = np.asarray(values)
        array = np.zeros(shape, dtype=values.dtype)
        array.put(indices, values)
        from .numpy import NumpyQTensor
        return NumpyQTensor(spaces, array)

    def __copy__(self):
        spaces = self._spaces
        values = self._values
        return SparseQTensor(spaces, values)

    def __repr__(self):
        spaces = tuple(self.spaces)
        values = self._values
        return f"{self.__class__.__name__}(spaces={repr(spaces)}, values={repr(values)})"

    # scalar operations

    @staticmethod
    def from_scalar(scalar):
        return SparseQTensor(tuple(), ((tuple(), scalar),))

    # linear operations

    def _formal_add(self, other):
        if not isinstance(other, SparseQTensor):
            return self.to_dense() + other

        new_spaces = self._spaces
        self_axes = tuple(other._spaces.index(space) for space in new_spaces)

        def iter_new_values():
            for coordinate, value in self._values.items():
                yield coordinate, value
            for other_coordinate, other_value in other._values.items():
                coordinate = tuple(other_coordinate[axis] for axis in self_axes)
                value = other_value
                yield coordinate, value

        new_values = iter_new_values()
        return SparseQTensor(new_spaces, new_values)

    def _formal_mul(self, other):
        new_spaces = self._spaces
        new_values = ((coordinate, value * other) for coordinate, value in self._values.items())
        return SparseQTensor(new_spaces, new_values)

    # tensor operations

    @property
    def ct(self):
        new_spaces = tuple(space.ct for space in self._spaces if isinstance(space, HSpace))
        new_values = ((coordinate, np.conjugate(value)) for coordinate, value in self._values.items())
        return SparseQTensor(new_spaces, new_values)

    def _formal_trace(self, *spaces: KetSpace):
        axes_ket = tuple(self._spaces.index(space) for space in spaces)
        axes_bra = tuple(self._spaces.index(space.ct) for space in spaces)
        axes_others = tuple(i for i in range(len(self._spaces)) if i not in axes_ket and i not in axes_bra)

        def iter_new_values():
            for coordinate, value in self._values.items():
                if all(coordinate[axis_ket] == coordinate[axis_bra] for axis_ket, axis_bra in zip(axes_ket, axes_bra)):
                    new_coordinate = tuple(coordinate[axis] for axis in axes_others)
                    new_value = value
                    yield new_coordinate, new_value

        new_spaces = tuple(self._spaces[axis] for axis in axes_others)
        new_values = iter_new_values()
        return SparseQTensor(new_spaces, new_values)

    def _formal_matmul(self, other):
        self_spaces = self._spaces

        from bnk import NumpyQTensor
        if not isinstance(other, (NumpyQTensor, SparseQTensor)):
            raise TypeError(f"Unsupported type {type(other)}")

        other_spaces = tuple(other._spaces)

        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_space in enumerate(self._spaces):
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
                for self_coordinate, self_value in self._values.items():
                    for other_coordinate, other_value in other._values.items():
                        if all(self_coordinate[axis_bra] == other_coordinate[axis_ket]
                               for axis_bra, axis_ket in zip(self_dot_axes, other_dot_axes)):
                            new_coordinate = tuple(self_coordinate[axis] for axis in new_self_axes) + \
                                             tuple(other_coordinate[axis] for axis in new_other_axes)
                            new_value = self_value * other_value
                            yield new_coordinate, new_value

            new_values = iter_new_values()
        elif isinstance(other, NumpyQTensor):
            other_values = other._values

            def iter_other_values(self_coordinate):
                other_new_coordinates = np.mgrid[tuple(slice(other_spaces[axis].n) for axis in new_other_axes)]
                other_new_coordinates = np.reshape(np.stack(other_new_coordinates, axis=-1), [-1, len(new_other_axes)])
                other_coordinates = np.zeros([len(other_new_coordinates), len(other_spaces)])
                other_coordinates[new_other_axes, :] = other_new_coordinates
                other_coordinates[other_dot_axes, :] = np.expand_dims(self_coordinate, -1)
                for other_coordinate in other_coordinates:
                    yield other_coordinate, other_values[tuple(other_coordinate)]

            def iter_new_values():
                for self_coordinate, self_value in self._values.items():
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
        from ..math import prod
        broadcast_identity = prod(ket_space.identity() for ket_space in ket_spaces)
        new_tensor = self @ broadcast_identity

        num_spaces = tuple(num_spaces)
        if len(num_spaces) > 0:
            raise TypeError("NumSpace is not supported yet!")

        return new_tensor
