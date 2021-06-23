import abc

from .numpy import NumpyMixin, NumpyPsiMixin, NumpyRhoMixin
from ..abstract import StaticMixin, StaticSteppingMixin


# mixin

class NumpyStaticSchrodingerMixin(StaticMixin, NumpyMixin, abc.ABC):
    @classmethod
    def init_model(cls, model, value):
        (hb, hmt, _), value, wrapping = super().init_model(model, value)
        return (hb, hmt), value, wrapping


class NumpyStaticSchrodingerSteppingMixin(NumpyStaticSchrodingerMixin, StaticSteppingMixin, abc.ABC):
    pass


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


class NumpyStaticSchrodingerPsiEuler(NumpyStaticSchrodingerSteppingMixin, NumpyPsiMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, psi, n, dt, **kwargs):
        hb, hmt = model
        kt = (dt / 1j / hb)

        for i in range(n):
            psi += kt * (hmt @ psi)

        return psi


class NumpyStaticSchrodingerRhoEuler(NumpyStaticSchrodingerSteppingMixin, NumpyRhoMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, rho, n, dt, **kwargs):
        hb, hmt = model
        kt = (dt / 1j / hb)

        for i in range(n):
            rho += kt * (hmt @ rho - rho @ hmt)

        return rho


class NumpyStaticSchrodingerPsiRk4(NumpyStaticSchrodingerSteppingMixin, NumpyPsiMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, psi, n, dt, **kwargs):
        hb, hmt = model
        k = -1j / hb
        dt2 = dt / 2.0

        for i in range(n):
            k1 = k * (hmt @ psi)
            k2 = k * (hmt @ (psi + dt2 * k1))
            k3 = k * (hmt @ (psi + dt2 * k2))
            k4 = k * (hmt @ (psi + dt * k3))
            psi += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return psi


class NumpyStaticSchrodingerRhoRk4(NumpyStaticSchrodingerSteppingMixin, NumpyRhoMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, rho, n, dt, **kwargs):
        hb, hmt = model
        k = -1j / hb
        dt2 = dt / 2.0

        for i in range(n):
            k1 = k * (hmt @ rho - rho @ hmt)
            rho1 = rho + dt2 * k1
            k2 = k * (hmt @ rho1 - rho1 @ hmt)
            rho2 = rho + dt2 * k2
            k3 = k * (hmt @ rho2 - rho2 @ hmt)
            rho3 = rho + dt * k3
            k4 = k * (hmt @ rho3 - rho3 @ hmt)
            rho += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return rho
