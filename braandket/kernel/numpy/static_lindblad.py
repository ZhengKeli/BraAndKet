import abc

import numpy as np

from .numpy import NumpyRhoMixin
from ..abstract import StaticStepping


# mixins

class NumpyStaticLindbladSteppingMixin(StaticStepping, NumpyRhoMixin, abc.ABC):
    @classmethod
    def init_model(cls, model, value):
        (hb, hmt, deco), value, wrapping = super().init_model(model, value)

        k_sh = - 1j / hb
        gamma_list, deco_list = zip(*deco)
        deco_ct_list = tuple(np.conj(np.transpose(deco)) for deco in deco_list)
        deco_ct_deco_list = tuple(deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list))

        return (k_sh, hmt, gamma_list, deco_list, deco_ct_list, deco_ct_deco_list), value, wrapping


# kernels

class NumpyStaticLindbladEulerKernel(NumpyStaticLindbladSteppingMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, rho, n, dt, **kwargs):
        k_sh, hmt, gamma_list, deco_list, deco_ct_list, deco_ct_deco_list = model

        for i in range(n):
            sh_part = hmt @ rho - rho @ hmt
            ln_part = np.sum([
                gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
                for gamma, deco, deco_ct, deco_ct_deco
                in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
            ], axis=0)
            rho += dt * (k_sh * sh_part + ln_part)

        return rho


class NumpyStaticLindbladRk4Kernel(NumpyStaticLindbladSteppingMixin):
    def __init__(self, model, time, value, *, dt=None):
        super().__init__(model, time, value, dt=dt)

    @classmethod
    def compute_static_stepping(cls, model, rho, n, dt, **kwargs):
        k_sh, hmt, gamma_list, deco_list, deco_ct_list, deco_ct_deco_list = model

        def d_rho(rho):
            sh_part = hmt @ rho - rho @ hmt
            ln_part = np.sum([
                gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
                for gamma, deco, deco_ct, deco_ct_deco
                in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
            ], axis=0)
            return k_sh * sh_part + ln_part

        dt2 = dt / 2.0
        for i in range(n):
            k1 = d_rho(rho)
            k2 = d_rho(rho + dt2 * k1)
            k3 = d_rho(rho + dt2 * k2)
            k4 = d_rho(rho + dt * k3)
            rho += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return rho
