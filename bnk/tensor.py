import numpy as np

from bnk.dimension import BraDimension, KetDimension
from bnk.utils import prod, unique

ZERO = object()


class Tensor:
    def __init__(self, dims, values=None):
        dims = tuple(dims)
        values = np.array(values)
        
        if not unique(dims):
            raise RuntimeError("There are duplicates of dims!")
        
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
    
    def transposed(self, new_dims):
        new_axes = [self.dims.index(new_dim) for new_dim in new_dims]
        new_values = np.transpose(self.values, axes=new_axes)
        return Tensor(new_dims, new_values)
    
    @property
    def flattened_values(self):
        ket_dims = (dim for dim in self.dims if isinstance(dim, KetDimension))
        ket_dims = sorted(ket_dims, key=lambda dim: (dim.key, dim.name))
        ket_flattened_dim = prod([dim.n for dim in ket_dims])
        bra_dims = (dim for dim in self.dims if isinstance(dim, BraDimension))
        bra_dims = sorted(bra_dims, key=lambda dim: (dim.key, dim.name))
        bra_flattened_dim = prod([dim.n for dim in bra_dims])
        transposed = self.transposed([*ket_dims, *bra_dims])
        flattened_values = np.reshape(transposed.values, [ket_flattened_dim, bra_flattened_dim])
        return flattened_values
    
    def trace(self, ket_dim: KetDimension):
        ket_axis = self.dims.index(ket_dim)
        if ket_axis == -1:
            return self
        
        bra_axis = self.dims.index(ket_dim.ct)
        if bra_axis == -1:
            raise TypeError("Not found matching BraDimension!")
        
        new_dims = tuple(dim for axis, dim in enumerate(self.dims) if axis not in (ket_axis, bra_axis))
        new_values = np.trace(self.values, axis1=ket_axis, axis2=bra_axis)
        
        return Tensor(new_dims, new_values)
    
    # tensor operations
    
    @property
    def ct(self):
        new_dims = [dim.ct for dim in self.dims]
        new_values = np.conjugate(self.values)
        return Tensor(new_dims, new_values)
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise NotImplementedError()
        
        self_dot_axes = []
        other_dot_axes = []
        for self_axis, self_dim in enumerate(self.dims):
            if not isinstance(self_dim, BraDimension):
                continue
            for other_axis, other_dim in enumerate(other.dims):
                if not isinstance(other_dim, KetDimension):
                    continue
                if self_dim.ket is not other_dim:
                    continue
                self_dot_axes.append(self_axis)
                other_dot_axes.append(other_axis)
                break
        
        new_self_dims = [dim for axis, dim in enumerate(self.dims) if axis not in self_dot_axes]
        new_other_dims = [dim for axis, dim in enumerate(other.dims) if axis not in other_dot_axes]
        
        new_dims = [*new_self_dims, *new_other_dims]
        new_values = np.tensordot(self.values, other.values, (self_dot_axes, other_dot_axes))
        
        return Tensor(new_dims, new_values)
    
    # linear operations
    
    def __neg__(self):
        return (-1) * self
    
    def __pos__(self):
        return self
    
    def __add__(self, other):
        if other is ZERO:
            return self
        if not isinstance(other, Tensor):
            raise NotImplementedError
        
        other = other.transposed(self.dims)
        new_dims = self.dims
        new_values = self.values + other.values
        return Tensor(new_dims, new_values)
    
    def __radd__(self, other):
        if other is ZERO:
            return self
        raise NotImplementedError()
    
    def __sub__(self, other):
        if other is ZERO:
            return self
        if not isinstance(other, Tensor):
            raise NotImplementedError
        other = other.transposed(self.dims)
        new_dims = self.dims
        new_values = self.values - other.values
        return Tensor(new_dims, new_values)
    
    def __rsub__(self, other):
        if other is ZERO:
            return -self
        raise NotImplementedError()
    
    def __mul__(self, other):
        new_dims = self.dims
        new_values = self.values * other
        return Tensor(new_dims, new_values)
    
    def __truediv__(self, other):
        new_dims = self.dims
        new_values = self.values / other
        return Tensor(new_dims, new_values)
    
    def __rmul__(self, other):
        return self * other
    
    def __float__(self):
        if len(self.dims) == 0:
            return np.abs(self.values)
        else:
            raise RuntimeError("Can not convert Tensor to float!")
