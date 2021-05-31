from .abstract import QTensor
from .numpy import NumpyQTensor
from .sparse import SparseQTensor

zero = SparseQTensor.from_scalar(0)
one = SparseQTensor.from_scalar(1)
