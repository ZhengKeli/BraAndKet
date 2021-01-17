from typing import Iterable

import numpy as np

from bnk.space import Space, HSpace, NumSpace, BraSpace, KetSpace


class QTensor:
    def __init__(self, dims: Iterable[Space], values):
        dims = tuple(dims)
        values = np.asarray(values)

        if len(set(dims)) != len(dims):
            raise ValueError("There are duplicated dims!")

        for dim, shape_n in zip(dims, np.shape(values)):
            if shape_n == dim.n:
                continue
            if isinstance(dim, NumSpace) and shape_n == 1:
                continue
            raise ValueError(f"The dim {dim} does not match the shape!")

        self._dims = dims
        self._values = values

    @property
    def dims(self):
        return self._dims

    @property
    def values(self):
        return self._values

    @property
    def dtype(self):
        return self.values.dtype

    # dimension operations

    def transposed(self, new_dims: Iterable[Space]):
        if self.dims == new_dims:
            return self
        new_axes = [self.dims.index(new_dim) for new_dim in new_dims]
        new_values = np.transpose(self.values, axes=new_axes)
        return QTensor(new_dims, new_values)

    def broadcast(self, broadcast_dims: Iterable[Space]):
        if len(self.dims) == 0:
            if self.values == 0:
                values = np.zeros([dim.n for dim in broadcast_dims])
                return QTensor(broadcast_dims, values)

        broadcast_num_dims = []
        broadcast_identity = one

        broadcast_dims = set(broadcast_dims)
        while broadcast_dims:
            broadcast_dim = broadcast_dims.pop()

            if broadcast_dim in self.dims:
                continue

            if isinstance(broadcast_dim, NumSpace):
                broadcast_num_dims.append(broadcast_dim)
                continue

            if isinstance(broadcast_dim, HSpace):
                if broadcast_dim.ct not in broadcast_dims:
                    raise TypeError(f"Can not broadcast unpaired dimension {broadcast_dim}.")
                broadcast_dims.remove(broadcast_dim.ct)
                broadcast_identity @= broadcast_dim.identity(self.dtype)
                continue

            raise TypeError(f"Unsupported custom type {type(broadcast_dim)} as a dimension type.")

        new_tensor = self @ broadcast_identity

        if broadcast_num_dims:
            num_shape = [1] * len(broadcast_num_dims)
            new_shape = (*num_shape, *np.shape(new_tensor.values))
            new_values = np.reshape(new_tensor.values, new_shape)
            broadcast_dims = (*broadcast_num_dims, *new_tensor.dims)
            new_tensor = QTensor(broadcast_dims, new_values)

        return new_tensor

    @staticmethod
    def wrap(flattened_values, flattened_dims, dims=None, copy=True):
        wrapped_dims = [dim for group in flattened_dims for dim in group]
        wrapped_shape = [dim.n for dim in wrapped_dims]
        wrapped_values = np.reshape(flattened_values, wrapped_shape)

        if copy:
            wrapped_values = np.copy(wrapped_values)

        tensor = QTensor(wrapped_dims, wrapped_values)
        if dims:
            tensor = tensor.transposed(dims)

        return tensor

    def flatten(self):
        num_dims = []
        ket_dims = []
        bra_dims = []
        for dim in self.dims:
            if isinstance(dim, KetSpace):
                ket_dims.append(dim)
            elif isinstance(dim, BraSpace):
                bra_dims.append(dim)
            else:
                num_dims.append(dim)

        num_dims = tuple(sorted(num_dims, key=lambda dm: (dm.name, -dm.n, id(dm))))
        ket_dims = tuple(sorted(ket_dims, key=lambda dm: (dm.name, -dm.n, id(dm))))
        bra_dims = tuple(sorted(bra_dims, key=lambda dm: (dm.name, -dm.n, id(dm))))

        flattened_num_dim = np.prod([dim.n for dim in num_dims], dtype=int)
        flattened_ket_dim = np.prod([dim.n for dim in ket_dims], dtype=int)
        flattened_bra_dim = np.prod([dim.n for dim in bra_dims], dtype=int)

        if flattened_num_dim == 1:
            flattened_dims = ket_dims, bra_dims
            flattened_shape = [flattened_ket_dim, flattened_bra_dim]
        else:
            flattened_dims = num_dims, ket_dims, bra_dims
            flattened_shape = [flattened_num_dim, flattened_ket_dim, flattened_bra_dim]

        transposed = self.transposed([*num_dims, *ket_dims, *bra_dims])
        flattened_values = np.reshape(transposed.values, flattened_shape)

        return flattened_dims, flattened_values

    @property
    def flattened_values(self):
        _, flattened_values = self.flatten()
        return flattened_values

    # tensor operations

    @property
    def ct(self):
        new_dims = (dim.ct for dim in self.dims if isinstance(dim, HSpace))
        new_values = np.conjugate(self.values)
        return QTensor(new_dims, new_values)

    def trace(self, *spaces: HSpace):
        if len(spaces) == 0:
            spaces = self.dims

        traced = self
        for space in spaces:
            ket_axis = traced.dims.index(space.ket)
            bra_axis = traced.dims.index(space.bra)
            new_dims = tuple(dim for axis, dim in enumerate(traced.dims) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced.values, axis1=ket_axis, axis2=bra_axis)
            traced = QTensor(new_dims, new_values)

        return traced

    def __matmul__(self, other):
        if not isinstance(other, QTensor):
            other_values = np.asarray(other)
            other_shape = np.shape(other_values)
            if not other_shape:
                return self * other
            other_dims = [NumSpace(shape_n) for shape_n in other_shape]
            other_tensor = QTensor(other_dims, other_values)
            return self @ other_tensor

        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_dim in enumerate(self.dims):
            # KetSpace ignored
            if isinstance(self_dim, BraSpace):
                try:
                    other_axis = other.dims.index(self_dim.ct)
                    self_dot_axes.append(self_axis)
                    other_dot_axes.append(other_axis)
                except ValueError:
                    pass
            elif isinstance(self_dim, NumSpace):
                try:
                    other.dims.index(self_dim)
                    raise NotImplementedError(f"Found batch dim {self_dim}. Not implemented matmul for batch dim!")
                except ValueError:
                    pass

        new_self_dims = [dim for axis, dim in enumerate(self.dims) if axis not in self_dot_axes]
        new_other_dims = [dim for axis, dim in enumerate(other.dims) if axis not in other_dot_axes]

        new_dims = [*new_self_dims, *new_other_dims]
        new_values = np.tensordot(self.values, other.values, (self_dot_axes, other_dot_axes))

        return QTensor(new_dims, new_values)

    def __rmatmul__(self, other):
        return other * self

    # linear operations

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

        broadcast_self = self.broadcast(other.dims)
        broadcast_other = other.broadcast(self.dims)

        new_dims = broadcast_self.dims
        broadcast_other = broadcast_other.transposed(new_dims)

        new_values = broadcast_self.values + broadcast_other.values
        return QTensor(new_dims, new_values)

    def __radd__(self, other):
        if other == 0:
            return self
        raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

    def __sub__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

        broadcast_self = self.broadcast(other.dims)
        broadcast_other = other.broadcast(self.dims)

        new_dims = broadcast_self.dims
        broadcast_other = broadcast_other.transposed(new_dims)

        new_values = broadcast_self.values - broadcast_other.values
        return QTensor(new_dims, new_values)

    def __rsub__(self, other):
        if other == 0:
            return -self
        raise TypeError("QTensor can perform + and - only with matching QTensors or 0.")

    def __mul__(self, other):
        if isinstance(other, QTensor):
            raise TypeError("Please use matmul operator \"@\" for QTensors.")
        if len(np.shape(other)) > 0:
            raise TypeError("QTensor can perform * and / only with a scalar.")
        new_dims = self.dims
        new_values = self.values * other
        return QTensor(new_dims, new_values)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, QTensor):
            raise TypeError("Please use matmul operator \"@\" for QTensors.")
        if len(np.shape(other)) > 0:
            raise TypeError("QTensor can perform * and / only with a scalar.")
        new_dims = self.dims
        new_values = self.values / other
        return QTensor(new_dims, new_values)

    # value operations

    def __float__(self):
        if len(self.dims) == 0:
            return np.abs(self.values)
        raise ValueError("Can not convert Tensor with rank>0 to float!")

    def numpy(self):
        return self.values

    # id operations

    def __hash__(self):
        return hash((self.dims, self.values))

    def __eq__(self, other):
        if self is other:
            return True
        if other == 0:
            return len(self.dims) == 0 and self.values == 0
        if not isinstance(other, QTensor):
            return False

        if self.dims == other.dims:
            if self.values is other.values:
                return True
            if np.all(self.values == other.values):
                return True

        broadcast_self = self.broadcast(other.dims)
        broadcast_other = other.broadcast(self.dims)
        broadcast_other = broadcast_other.transposed(broadcast_self.dims)

        return np.all(broadcast_self.values == broadcast_other.values)


zero = QTensor([], np.zeros([], np.float))

one = QTensor([], np.ones([], np.float))
