import numpy as np
from tqdm import tqdm

from bnk.reduce import ReducedHSpace
from bnk.tensor import QTensor


# evolve functions

def schrodinger_evolve(
        t, psi, hmt, hb, span, dt=None,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt = hmt if isinstance(hmt, QTensor) else sum(hmt)

    # unwrap
    org_dims = psi.dims
    if reduce:
        reduction = ReducedHSpace.from_initial([psi], [hmt])
        psi = reduction.reduce(psi)
        hmt = reduction.reduce(hmt)
    else:
        reduction = None
        hmt = hmt.broadcast(psi.dims)

    flat_dims, psi = psi.flatten()
    hmt = hmt.flattened_values

    psi = np.asarray(psi, np.complex64)
    hmt = np.asarray(hmt, np.complex64)

    # compute
    evolve_func = get_schrodinger_evolve_func(hmt, hb, dt, method)

    evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, flat_dims, org_dims, reduction)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    psi = evolve_func(t, psi, span)
    t = t + span

    # wrap
    if reduce:
        psi = QTensor.wrap(psi, flat_dims)
        psi = reduction.inflate(psi)
        psi = psi.transposed(org_dims)
    else:
        psi = QTensor.wrap(psi, flat_dims, org_dims)

    return t, psi


def lindblad_evolve(
        t, rho, hmt, deco, gamma, hb, span, dt,
        dlt=None, log_func=None,
        reduce=True, rectify=True,
        verbose=True):
    hmt = hmt if isinstance(hmt, QTensor) else sum(hmt)
    deco_list = [deco] if isinstance(deco, QTensor) else list(deco)
    gamma_list = np.reshape(gamma, [len(deco_list)])

    # unwrap
    org_dims = rho.dims
    if reduce:
        reduction = ReducedHSpace.from_initial([rho], [hmt, *deco_list])
        rho = reduction.reduce(rho)
        hmt = reduction.reduce(hmt)
        deco_list = [reduction.reduce(deco) for deco in deco_list]
    else:
        reduction = None
        all_dims = {
            *rho.dims,
            *hmt.dims,
            *(dim for deco in deco_list for dim in deco.dims)
        }
        rho = rho.broadcast(all_dims)
        hmt = hmt.broadcast(all_dims)
        deco_list = [deco.broadcast(all_dims) for deco in deco_list]

    flat_dims, rho = rho.flatten()
    rho = np.asarray(rho, dtype=np.complex64)

    hmt = hmt.flattened_values
    hmt = np.asarray(hmt, dtype=np.complex64)

    deco_list = [deco.flattened_values for deco in deco_list]
    deco_list = [np.asarray(deco, dtype=np.complex64) for deco in deco_list]
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    # compute
    def steps_evolve_func(_, rho, n, dt):
        return lindblad_evolve_kernel(
            rho, hmt, gamma_list, deco_list, deco_ct_list, deco_ct_deco_list, hb, dt, n)

    evolve_func = get_tiling_steps_evolve_func(steps_evolve_func, dt)

    evolve_func = get_rectifying_rho_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, flat_dims, org_dims, reduction)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    rho = evolve_func(t, rho, span)
    t = t + span

    # wrap
    if reduce:
        rho = QTensor.wrap(rho, flat_dims)
        rho = reduction.inflate(rho)
        rho = rho.transposed(org_dims)
    else:
        rho = QTensor.wrap(rho, flat_dims, org_dims)

    return t, rho


def dynamic_schrodinger_evolve(t, psi, hmt, k_func, hb, span, dt,
                               dlt=None, log_func=None, rectify=True, verbose=True):
    hmt_list = [hmt] if isinstance(hmt, QTensor) else list(hmt)
    hmt_count = len(hmt_list)

    # broadcast
    hmt_list = [hmt.broadcast(psi.dims) for hmt in hmt_list]

    # unwrap
    dims = psi.dims
    flat_dims, psi = psi.flatten()
    psi = np.asarray(psi, dtype=np.complex64)

    hmt_list = [hmt.flattened_values for hmt in hmt_list]
    hmt_list = [np.asarray(hmt, dtype=np.complex64) for hmt in hmt_list]

    hmt_list = np.stack(hmt_list, -1)

    # compute
    def _evolve_func(t, psi, sp):
        n = int(np.ceil(sp / dt))
        fdt = sp / n

        kt = (fdt / 1j / hb)
        for i in range(n):
            ti = t + i * fdt

            k_list = np.reshape(k_func(ti), [hmt_count])
            hmt = np.sum(hmt_list * k_list, -1)

            psi += kt * (hmt @ psi)

        if rectify:
            psi /= np.sqrt(np.sum(np.conj(psi) * psi))
        return psi

    _log_func = get_wrapping_log_func(log_func, flat_dims, dims)

    evolve_func = get_logging_evolve_func(_evolve_func, _log_func, dlt, verbose)

    t, psi = evolve_func(t, psi, span)

    # wrap
    psi = QTensor.wrap(psi, flat_dims, dims)

    return t, psi


def dynamic_lindblad_evolve(t, rho, hmt, k_func, deco, gamma_func, hb, span, dt,
                            dlt=None, log_func=None, rectify=True, verbose=True):
    hmt_list = [hmt] if isinstance(hmt, QTensor) else list(hmt)
    deco_list = [deco] if isinstance(deco, QTensor) else list(deco)
    hmt_count = len(hmt_list)
    deco_count = len(deco_list)

    # broadcast
    all_dims = {
        *rho.dims,
        *(dim for hmt in hmt_list for dim in hmt.dims),
        *(dim for deco in deco_list for dim in deco.dims)
    }
    rho = rho.broadcast(all_dims)
    hmt_list = [hmt.broadcast(all_dims) for hmt in hmt_list]
    deco_list = [deco.broadcast(all_dims) for deco in deco_list]

    # unwrap
    dims = rho.dims
    flat_dims, rho = rho.flatten()
    rho = np.asarray(rho, dtype=np.complex64)

    hmt_list = [hmt.flattened_values for hmt in hmt_list]
    hmt_list = [np.asarray(hmt, dtype=np.complex64) for hmt in hmt_list]
    hmt_list = np.stack(hmt_list, -1)

    deco_list = [deco.flattened_values for deco in deco_list]
    deco_list = [np.asarray(deco, dtype=np.complex64) for deco in deco_list]
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    # compute
    def _evolve_func(t, rho, sp):
        n = int(np.ceil(sp / dt))
        fdt = sp / n

        k_sh = - 1j / hb
        for i in range(n):
            ti = t + i * fdt

            k_list = np.reshape(k_func(ti), [hmt_count])
            hmt = np.sum(hmt_list * k_list, -1)
            sh_part = hmt @ rho - rho @ hmt

            gamma_list = np.reshape(gamma_func(ti), [deco_count])
            ln_part = sum(
                gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
                for gamma, deco, deco_ct, deco_ct_deco
                in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list))

            rho += fdt * (k_sh * sh_part + ln_part)

        if rectify:
            rho /= np.trace(rho, axis1=-2, axis2=-1)
        return rho

    _log_func = get_wrapping_log_func(log_func, flat_dims, dims)

    evolve_func = get_logging_evolve_func(_evolve_func, _log_func, dlt, verbose)

    t, rho = evolve_func(t, rho, span)

    # wrap
    rho = QTensor.wrap(rho, flat_dims, dims)

    return t, rho


# kernel selection

def get_schrodinger_evolve_func(hmt, hb, dt, method):
    if method is None:
        try:
            from scipy.linalg import expm
            method = 'pade'
        except ImportError:
            method = 'euler'
    else:
        method = str(method).lower()

    if method == 'pade':
        def evolve_func(_, psi, span):
            return schrodinger_kernel_pade(psi, hmt, hb, span)
    else:
        if method == 'euler':
            def steps_evolve_func(_, psi, n, dt):
                return schrodinger_kernel_euler(psi, hmt, hb, dt, n)
        elif method == 'rk4':
            def steps_evolve_func(_, psi, n, dt):
                return schrodinger_kernel_rk4(psi, hmt, hb, dt, n)
        else:
            raise TypeError(f"Unsupported method {method}!")

        evolve_func = get_tiling_steps_evolve_func(steps_evolve_func, dt)
    return evolve_func


def schrodinger_kernel_pade(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, span):
    from scipy.linalg import expm
    psi = expm((span / 1j / hb) * hmt) @ psi
    return psi


def schrodinger_kernel_euler(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    kt = (dt / 1j / hb)
    for i in range(n):
        psi += kt * (hmt @ psi)
    return psi


def schrodinger_kernel_rk4(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    dt2 = dt / 2.0
    for i in range(n):
        k1 = (-1j / hb) * (hmt @ psi)
        k2 = (-1j / hb) * (hmt @ (psi + dt2 * k1))
        k3 = (-1j / hb) * (hmt @ (psi + dt2 * k2))
        k4 = (-1j / hb) * (hmt @ (psi + dt * k3))
        psi += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return psi


def lindblad_evolve_kernel(
        rho: np.ndarray, hmt: np.ndarray,
        gamma_list, deco_list, deco_ct_list, deco_ct_deco_list,
        hb, dt, n):
    k_sh = - 1j / hb
    for i in range(n):
        sh_part = hmt @ rho - rho @ hmt

        ln_part = sum(
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list))

        rho += dt * (k_sh * sh_part + ln_part)
    return rho


# utils

def get_logging_evolve_func(evolve_func, log_func=None, dlt=None, verbose=True):
    if dlt is None:
        return evolve_func

    def _logging_evolve_func(t, v, span):
        progress = None
        if log_func:
            log_func(t, v)
        if verbose:
            progress = tqdm(total=span)

        mt = t + span
        while True:
            rt = mt - t
            if rt < dlt:
                break

            v = evolve_func(t, v, dlt)
            t += dlt

            if log_func:
                log_func(t, v)
            if verbose:
                progress.update(dlt)

        if rt > 0:
            v = evolve_func(t, v, rt)
            t = mt

            if log_func:
                log_func(t, v)
            if verbose:
                progress.update(rt)

        return v

    return _logging_evolve_func


def get_wrapping_log_func(log_func, flat_dims, org_dims, reduction: ReducedHSpace = None):
    if log_func is None:
        return None

    if reduction is None:
        def _wrapping_log_func(t_log, psi_log):
            psi_log = QTensor.wrap(psi_log, flat_dims, org_dims)
            return log_func(t_log, psi_log)
    else:
        def _wrapping_log_func(t_log, psi_log):
            psi_log = QTensor.wrap(psi_log, flat_dims)
            psi_log = reduction.inflate(psi_log)
            psi_log = psi_log.transposed(org_dims)
            return log_func(t_log, psi_log)

    return _wrapping_log_func


def get_tiling_steps_evolve_func(steps_evolve_func, dt):
    def _tiling_evolve_func(t, v, span):
        tiled_n = int(np.ceil(span / dt))
        tiled_dt = span / tiled_n
        return steps_evolve_func(t, v, tiled_n, tiled_dt)

    return _tiling_evolve_func


def get_rectifying_psi_evolve_func(evolve_func, rectify=True):
    if not rectify:
        return evolve_func

    def _rectifying_evolve_func(t, psi, span):
        psi = evolve_func(t, psi, span)
        psi = rectify_psi(psi)
        return psi

    return _rectifying_evolve_func


def get_rectifying_rho_evolve_func(evolve_func, rectify=True):
    if not rectify:
        return evolve_func

    def _rectifying_evolve_func(t, rho, span):
        rho = evolve_func(t, rho, span)
        rho = rectify_rho(rho)
        return rho

    return _rectifying_evolve_func


def rectify_psi(psi):
    psi /= np.sqrt(np.sum(np.conj(psi) * psi))
    return psi


def rectify_rho(rho):
    rho /= np.trace(rho, axis1=-2, axis2=-1)
    return rho
