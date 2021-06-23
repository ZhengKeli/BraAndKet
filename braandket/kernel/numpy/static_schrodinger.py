import abc

from .numpy import NumpyMixin, NumpyPsiMixin, NumpyRhoMixin
from ..abstract import Static


# mixin

class NumpyStaticSchrodingerMixin(Static, NumpyMixin, abc.ABC):
    @classmethod
    def init_model(cls, model, value):
        (hb, hmt, _), value, wrapping = super().init_model(model, value)
        return (hb, hmt), value, wrapping


# kernels

class NumpyStaticSchrodingerPsiPadeKernel(NumpyStaticSchrodingerMixin, NumpyPsiMixin):
    def __init__(self, model, time, value):
        super().__init__(model, time, value)

    @classmethod
    def compute_static(cls, model, psi, span, **kwargs):
        hb, hmt = model

        from scipy.linalg import expm
        op = expm((span / 1j / hb) * hmt)
        return op @ psi


class NumpyStaticSchrodingerRhoPadeKernel(NumpyStaticSchrodingerMixin, NumpyRhoMixin):
    def __init__(self, model, time, value):
        super().__init__(model, time, value)

    @classmethod
    def compute_static(cls, model, rho, span, **kwargs):
        hb, hmt = model

        from scipy.linalg import expm
        op = expm((span / 1j / hb) * hmt)
        return op @ rho @ op
