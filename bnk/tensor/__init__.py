from .abstract import QTensor
from .numpy import NumpyQTensor

zero = NumpyQTensor.from_scalar(0.0)
one = NumpyQTensor.from_scalar(1.0)
