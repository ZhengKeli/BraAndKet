import numpy as np
from bnk.dimension import KetDimension
from bnk.tensor import Tensor


class Space(KetDimension):
    def eigenstate(self, index):
        values = np.zeros(self.n, np.float)
        values[index] = 1.0
        return Tensor([self], values)
    
    def zero_vector(self):
        return Tensor([self], np.zeros([self.n]))
    
    def identity_operator(self):
        return Operator([self], np.eye(self.n))
    
    def zero_operator(self):
        return Operator([self], np.zeros([self.n, self.n]))


def KetVector(space: Space, values):
    return Tensor(space, values)


def BraVector(space: Space, values):
    return KetVector(space, values).ct


def Operator(spaces, values):
    ket_dims = spaces
    bra_dims = [ket_dim.ct for ket_dim in ket_dims]
    return Tensor(ket_dims + bra_dims, values)
