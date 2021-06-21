import abc

import numpy as np

from .abstract import Kernel, Static, StaticStepping
from ..tensor import QTensor


# flattening mixin

class Flattening(Kernel, abc.ABC):

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


class FlatteningPsi(Flattening, abc.ABC):
    @classmethod
    def normalize(cls, value):
        return value / np.sum(value * np.conj(value))


class FlatteningRho(Flattening, abc.ABC):
    @classmethod
    def normalize(cls, value):
        return value / np.trace(value)


# schrodinger pade

class SchrodingerPsiPade(Static, FlatteningPsi):

    @classmethod
    def compute_static(cls, model, psi, span, **kwargs):
        hb, hmt, _ = model

        from scipy.linalg import expm
        op = expm((span / 1j / hb) * hmt)
        return op @ psi


class SchrodingerRhoPade(Static, FlatteningRho):
    @classmethod
    def compute_static(cls, model, rho, span, **kwargs):
        hb, hmt, _ = model

        from scipy.linalg import expm
        op = expm((span / 1j / hb) * hmt)
        return op @ rho @ op


# lindblad

class Lindblad(StaticStepping, FlatteningRho, abc.ABC):
    @classmethod
    def init_model(cls, model, value):
        (hb, hmt, deco), value, wrapping = super().init_model(model, value)

        k_sh = - 1j / hb
        gamma_list, deco_list = zip(*deco)
        deco_ct_list = tuple(np.conj(np.transpose(deco)) for deco in deco_list)
        deco_ct_deco_list = tuple(deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list))

        return (k_sh, hmt, gamma_list, deco_list, deco_ct_list, deco_ct_deco_list), value, wrapping


class LindbladEuler(Lindblad):
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


class LindbladRk4(Lindblad):
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
