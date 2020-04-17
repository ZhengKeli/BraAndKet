import enum
from typing import Iterable, Tuple

import numpy as np


# space

class HSpace:
    def __init__(self, n, name=None, key=None):
        self._n = n
        self._name = name
        self._key = key
    
    @property
    def n(self):
        return self._n
    
    @property
    def name(self):
        return self._name
    
    @property
    def key(self):
        return self._key
    
    def __repr__(self):
        return f"<Space: n={self.n}, name={self.name}, id={id(self)}>"
    
    def eigenstate(self, index):
        values = np.zeros(self.n, np.float)
        values[index] = 1.0
        return KetVector([self], values)
    
    def identity(self):
        return Operator([self], np.eye(self.n))


# dimension

class DimensionType(enum.Enum):
    Other = enum.auto()
    Ket = enum.auto()
    Bra = enum.auto()
    
    @property
    def ct(self):
        if self is DimensionType.Ket:
            return DimensionType.Bra
        elif self is DimensionType.Bra:
            return DimensionType.Ket
        else:
            return None
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)


class QDimension:
    def __init__(self, typ: DimensionType, space: HSpace):
        self._typ = typ
        self._space = space
    
    @property
    def typ(self):
        return self._typ
    
    @property
    def is_ket(self):
        return self.typ == DimensionType.Ket
    
    @property
    def is_bra(self):
        return self.typ == DimensionType.Bra
    
    @property
    def space(self):
        return self._space
    
    @property
    def n(self):
        return self.space.n
    
    @property
    def name(self):
        return self.space.name
    
    @property
    def key(self):
        return self.space.key
    
    @property
    def ct(self):
        return QDimension(self.typ.ct, self.space)
    
    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, QDimension) and \
               other.space == self.space and \
               other.typ == self.typ
    
    def __repr__(self):
        return f"<Dimension: typ={self.typ}, n={self.n}, name={self.name}>"


def KetDimension(space: HSpace):
    return QDimension(DimensionType.Ket, space)


def BraDimension(space: HSpace):
    return QDimension(DimensionType.Bra, space)


def OtherDimension(space: HSpace):
    return QDimension(DimensionType.Other, space)


# tensor

class QTensor:
    def __init__(self, dims: Iterable[QDimension], values):
        dims: Tuple[QDimension, ...] = tuple(dims)
        values: np.ndarray = np.asarray(values)
        
        for i, dim in enumerate(dims):
            if any(dim == dim2 for dim2 in dims[i + 1:]):
                raise RuntimeError("There are duplicated dims!")
        
        if tuple(dim.n for dim in dims) != np.shape(values):
            raise RuntimeError("The shape of values does not match the dims!")
        
        self._dims = dims
        self._values = values
    
    @property
    def dims(self):
        return self._dims
    
    @property
    def values(self):
        return self._values
    
    # dimension operations
    
    def transposed(self, new_dims: Iterable[QDimension]):
        new_axes = [self.dims.index(new_dim) for new_dim in new_dims]
        new_values = np.transpose(self.values, axes=new_axes)
        return QTensor(new_dims, new_values)
    
    def broadcast(self, new_dims: Iterable[QDimension]):
        if len(self.dims) == 0:
            if self.values == 0:
                values = np.zeros([dim.n for dim in new_dims])
                return QTensor(new_dims, values)
        
        broadcast_pairs = {}
        for dim in new_dims:
            if dim in self.dims:
                continue
            pair = broadcast_pairs.get(dim.space)
            if pair is None:
                pair = []
                broadcast_pairs[dim.space] = pair
            pair.append(dim)
        
        broadcast_identity = one
        for space, dims in broadcast_pairs.items():
            if len(dims) != 2 or dims[0] != dims[1].ct:
                raise TypeError("Can only broadcast on the paired dimensions.")
            broadcast_identity @= space.identity()
        
        return self @ broadcast_identity
    
    def trace(self, *spaces: HSpace):
        if len(spaces) == 0:
            spaces = tuple(dim.space for dim in self.dims)
        
        traced = self
        for space in spaces:
            try:
                ket_axis = traced.dims.index(KetDimension(space))
            except ValueError:
                continue
            try:
                bra_axis = traced.dims.index(BraDimension(space))
            except ValueError:
                continue
            
            new_dims = tuple(dim for axis, dim in enumerate(traced.dims) if axis not in (ket_axis, bra_axis))
            new_values = np.trace(traced.values, axis1=ket_axis, axis2=bra_axis)
            traced = QTensor(new_dims, new_values)
        
        return traced
    
    def flatten(self):
        oth_dims = []
        ket_dims = []
        bra_dims = []
        for dim in self.dims:
            if dim.is_ket:
                ket_dims.append(dim)
            elif dim.is_bra:
                bra_dims.append(dim)
            else:
                oth_dims.append(dim)
        
        oth_dims = tuple(sorted(oth_dims, key=lambda dm: (dm.key, -dm.n, dm.name)))
        ket_dims = tuple(sorted(ket_dims, key=lambda dm: (dm.key, -dm.n, dm.name)))
        bra_dims = tuple(sorted(bra_dims, key=lambda dm: (dm.key, -dm.n, dm.name)))
        
        flattened_oth_dim = np.prod([dim.n for dim in oth_dims], dtype=int)
        flattened_ket_dim = np.prod([dim.n for dim in ket_dims], dtype=int)
        flattened_bra_dim = np.prod([dim.n for dim in bra_dims], dtype=int)
        
        if flattened_oth_dim == 1:
            corr_dims = ket_dims, bra_dims
            flattened_shape = [flattened_ket_dim, flattened_bra_dim]
        else:
            corr_dims = oth_dims, ket_dims, bra_dims
            flattened_shape = [flattened_oth_dim, flattened_ket_dim, flattened_bra_dim]
        
        transposed = self.transposed([*oth_dims, *ket_dims, *bra_dims])
        flattened_values = np.reshape(transposed.values, flattened_shape)
        
        return corr_dims, flattened_values
    
    @property
    def flattened_values(self):
        _, flattened_values = self.flatten()
        return flattened_values
    
    # tensor operations
    
    @property
    def ct(self):
        new_dims = [dim.ct for dim in self.dims]
        new_values = np.conjugate(self.values)
        return QTensor(new_dims, new_values)
    
    def __matmul__(self, other):
        if not isinstance(other, QTensor):
            raise NotImplementedError()
        
        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_dim in enumerate(self.dims):
            if not self_dim.is_bra:
                continue
            for other_axis, other_dim in enumerate(other.dims):
                if not other_dim.is_ket:
                    continue
                if self_dim.space != other_dim.space:
                    continue
                self_dot_axes.append(self_axis)
                other_dot_axes.append(other_axis)
                break
        
        new_self_dims = [dim for axis, dim in enumerate(self.dims) if axis not in self_dot_axes]
        new_other_dims = [dim for axis, dim in enumerate(other.dims) if axis not in other_dot_axes]
        
        new_dims = [*new_self_dims, *new_other_dims]
        new_values = np.tensordot(self.values, other.values, (self_dot_axes, other_dot_axes))
        
        return QTensor(new_dims, new_values)
    
    # linear operations
    
    def __neg__(self):
        return (-1) * self
    
    def __pos__(self):
        return self
    
    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise NotImplementedError
        broadcast_self = self.broadcast(other.dims)
        broadcast_other = other.broadcast(self.dims)
        
        new_dims = broadcast_self.dims
        broadcast_other = broadcast_other.transposed(new_dims)
        
        new_values = broadcast_self.values + broadcast_other.values
        return QTensor(new_dims, new_values)
    
    def __radd__(self, other):
        if other == 0:
            return self
        raise NotImplementedError()
    
    def __sub__(self, other):
        if other == 0:
            return self
        if not isinstance(other, QTensor):
            raise NotImplementedError
        broadcast_self = self.broadcast(other.dims)
        broadcast_other = other.broadcast(self.dims)
        
        new_dims = broadcast_self.dims
        broadcast_other = broadcast_other.transposed(new_dims)
        
        new_values = broadcast_self.values - broadcast_other.values
        return QTensor(new_dims, new_values)
    
    def __rsub__(self, other):
        if other == 0:
            return -self
        raise NotImplementedError()
    
    def __mul__(self, other):
        new_dims = self.dims
        new_values = self.values * other
        return QTensor(new_dims, new_values)
    
    def __truediv__(self, other):
        new_dims = self.dims
        new_values = self.values / other
        return QTensor(new_dims, new_values)
    
    def __rmul__(self, other):
        return self * other
    
    # other operations
    
    def __float__(self):
        if len(self.dims) == 0:
            return np.abs(self.values)
        else:
            raise RuntimeError("Can not convert Tensor with rank>0 to float!")
    
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


def KetVector(spaces: Iterable[HSpace], values):
    ket_dims = [KetDimension(space) for space in spaces]
    return QTensor(ket_dims, values)


def BraVector(spaces: Iterable[HSpace], values):
    bra_dims = [BraDimension(space) for space in spaces]
    return QTensor(bra_dims, values)


def Operator(spaces: Iterable[HSpace], values):
    ket_dims = [KetDimension(space) for space in spaces]
    bra_dims = [BraDimension(space) for space in spaces]
    return QTensor(ket_dims + bra_dims, values)
