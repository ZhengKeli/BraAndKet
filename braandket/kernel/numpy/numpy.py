import abc

import numpy as np

from ..abstract import Kernel
from ...tensor import QTensor


class NumpyMixin(Kernel, abc.ABC):

    @classmethod
    def init_model(cls, model, value):
        hb, hmt, deco, dynamic_hmt, dynamic_deco = model

        spaces = value.spaces

        hmt = sum(k * h for k, h in hmt)
        hmt = hmt.broadcast(spaces).flatten(dtype=np.complex64)

        deco = tuple(
            (gamma, de.broadcast(spaces).flatten(dtype=np.complex64))
            for gamma, de in deco)

        dynamic_hmt = tuple(
            (k_func, h.broadcast(spaces).flatten(dtype=np.complex64))
            for k_func, h in dynamic_hmt)

        dynamic_deco = tuple(
            (gamma_func, de.broadcast(spaces).flatten(dtype=np.complex64))
            for gamma_func, de in dynamic_deco)

        value, *wrapping = value.flatten(dtype=np.complex64, return_spaces=True)

        return (hb, hmt, deco, dynamic_hmt, dynamic_deco), value, wrapping

    @classmethod
    def unwrap_value(cls, value, wrapping):
        return value.flatten(*wrapping, dtype=np.complex64)

    @classmethod
    def wrap_value(cls, value, wrapping):
        return QTensor.inflate(value, *wrapping)


class NumpyPsiMixin(NumpyMixin, abc.ABC):
    @classmethod
    def normalize(cls, value):
        return value / np.sum(value * np.conj(value))


class NumpyRhoMixin(NumpyMixin, abc.ABC):
    @classmethod
    def normalize(cls, value):
        return value / np.trace(value)
