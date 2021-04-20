import numpy as np
from tqdm import tqdm

from ..pruning import PrunedKetSpace
from ..tensor import QTensor
from ..utils import structured_iter, structured_map


# evolve functions

def schrodinger_evolve(
        t, psi_or_rho, hmt, hb, span, dt=None,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt = hmt if isinstance(hmt, QTensor) else sum(hmt)
    is_rho = psi_or_rho.is_rho

    # unwrap
    psi_or_rho, hmt, wrapping = unwrap(psi_or_rho, hmt, reduce=reduce)

    # compute
    evolve_func = get_schrodinger_evolve_func(hmt, hb, dt, method, is_rho)

    if not is_rho:
        evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)
    else:
        evolve_func = get_rectifying_rho_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    psi_or_rho = evolve_func(t, psi_or_rho, span)
    t = t + span

    # wrap
    psi_or_rho = wrap(psi_or_rho, wrapping)

    return t, psi_or_rho


def lindblad_evolve(
        t, rho, hmt, deco, gamma, hb, span, dt,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt = hmt if isinstance(hmt, QTensor) else sum(hmt)
    deco_list = [deco] if isinstance(deco, QTensor) else list(deco)
    gamma_list = np.reshape(gamma, [len(deco_list)])

    # unwrap
    rho, (hmt, deco_list), wrapping = unwrap(rho, (hmt, deco_list), reduce=reduce)

    # compute
    evolve_func = get_lindblad_evolve_func(hmt, deco_list, gamma_list, hb, dt, method)

    evolve_func = get_rectifying_rho_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    rho = evolve_func(t, rho, span)
    t = t + span

    # wrap
    rho = wrap(rho, wrapping)

    return t, rho


def dynamic_schrodinger_evolve(
        t, psi_or_rho, hmt, k_func, hb, span, dt,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt_list = [hmt] if isinstance(hmt, QTensor) else list(hmt)
    hmt_count = len(hmt_list)
    is_rho = psi_or_rho.is_rho

    # unwrap
    psi_or_rho, hmt_list, wrapping = unwrap(psi_or_rho, hmt_list, reduce=reduce)

    # prepare
    hmt_list = np.stack(hmt_list, -1)

    # compute
    if not is_rho:
        def dv(t, psi):
            k_list = np.reshape(k_func(t), [hmt_count])
            hmt = np.sum(hmt_list * k_list, -1)
            return -1j / hb * (hmt @ psi)
    else:
        def dv(t, rho):
            k_list = np.reshape(k_func(t), [hmt_count])
            hmt = np.sum(hmt_list * k_list, -1)
            return -1j / hb * (hmt @ rho @ hmt)

    stepping_kernel = get_differential_stepping_kernel(dv, method)

    evolve_func = get_stepping_evolve_func(stepping_kernel, dt)

    if not is_rho:
        evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)
    else:
        evolve_func = get_rectifying_rho_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    psi_or_rho = evolve_func(t, psi_or_rho, span)
    t = t + span

    # wrap
    psi_or_rho = wrap(psi_or_rho, wrapping)

    return t, psi_or_rho


def dynamic_lindblad_evolve(
        t, rho, hmt, k_func, deco, gamma_func, hb, span, dt,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt_list = [hmt] if isinstance(hmt, QTensor) else list(hmt)
    deco_list = [deco] if isinstance(deco, QTensor) else list(deco)
    hmt_count = len(hmt_list)
    deco_count = len(deco_list)

    # unwrap
    rho, (hmt_list, deco_list), wrapping = unwrap(rho, (hmt_list, deco_list), reduce=reduce)

    # prepare
    hmt_list = np.stack(hmt_list, -1)
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    # compute
    def dv(t, rho):
        k_list = np.reshape(k_func(t), [hmt_count])
        hmt = np.sum(hmt_list * k_list, -1)
        sh_part = hmt @ rho - rho @ hmt

        gamma_list = np.reshape(gamma_func(t), [deco_count])
        ln_part = sum([
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
        ])

        return (- 1j / hb) * sh_part + ln_part

    stepping_kernel = get_differential_stepping_kernel(dv, method)

    evolve_func = get_stepping_evolve_func(stepping_kernel, dt)

    evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    rho = evolve_func(t, rho, span)
    t = t + span

    # wrap
    rho = wrap(rho, wrapping)

    return t, rho


# kernels: schrodinger

def get_schrodinger_evolve_func(hmt, hb, dt, method, is_rho):
    if method is None:
        try:
            from scipy.linalg import expm
            method = 'pade'
        except ImportError:
            method = 'euler'
    else:
        method = str(method).lower()

    if method == 'pade':
        if not is_rho:
            def evolve_func(_, psi, span):
                return schrodinger_kernel_pade(psi, hmt, hb, span)
        else:
            def evolve_func(_, rho, span):
                return schrodinger_kernel_pade_rho(rho, hmt, hb, span)
    else:
        if method == 'euler':
            if not is_rho:
                def stepping_kernel(_, psi, n, dt):
                    return schrodinger_kernel_euler(psi, hmt, hb, dt, n)
            else:
                def stepping_kernel(_, rho, n, dt):
                    return schrodinger_kernel_euler_rho(rho, hmt, hb, dt, n)
        elif method == 'rk4':
            if not is_rho:
                def stepping_kernel(_, psi, n, dt):
                    return schrodinger_kernel_rk4(psi, hmt, hb, dt, n)
            else:
                def stepping_kernel(_, rho, n, dt):
                    return schrodinger_kernel_rk4_rho(rho, hmt, hb, dt, n)
        else:
            raise TypeError(f"Unsupported method {method}!")

        evolve_func = get_stepping_evolve_func(stepping_kernel, dt)
    return evolve_func


def schrodinger_kernel_pade(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, span):
    from scipy.linalg import expm
    op = expm((span / 1j / hb) * hmt)
    return op @ psi


def schrodinger_kernel_pade_rho(
        rho: np.ndarray,
        hmt: np.ndarray,
        hb, span):
    from scipy.linalg import expm
    op = expm((span / 1j / hb) * hmt)
    return op @ rho @ op


def schrodinger_kernel_euler(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    kt = (dt / 1j / hb)
    for i in range(n):
        psi += kt * (hmt @ psi)
    return psi


def schrodinger_kernel_euler_rho(
        rho: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    kt = (dt / 1j / hb)
    for i in range(n):
        rho += kt * (hmt @ rho @ hmt)
    return rho


def schrodinger_kernel_rk4(
        psi: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    k = -1j / hb
    dt2 = dt / 2.0
    for i in range(n):
        k1 = k * (hmt @ psi)
        k2 = k * (hmt @ (psi + dt2 * k1))
        k3 = k * (hmt @ (psi + dt2 * k2))
        k4 = k * (hmt @ (psi + dt * k3))
        psi += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return psi


def schrodinger_kernel_rk4_rho(
        rho: np.ndarray,
        hmt: np.ndarray,
        hb, dt, n):
    k = -1j / hb
    dt2 = dt / 2.0
    for i in range(n):
        k1 = k * (hmt @ rho @ hmt)
        k2 = k * (hmt @ (rho + dt2 * k1) @ hmt)
        k3 = k * (hmt @ (rho + dt2 * k2) @ hmt)
        k4 = k * (hmt @ (rho + dt * k3) @ hmt)
        rho += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return rho


# kernels: lindblad

def get_lindblad_evolve_func(hmt, deco_list, gamma_list, hb, dt, method):
    if method is None:
        method = 'euler'
    else:
        method = str(method).lower()

    if method == 'euler':
        def stepping_kernel(_, rho, n, dt):
            return lindblad_kernel_euler(rho, hmt, gamma_list, deco_list, hb, dt, n)

        evolve_func = get_stepping_evolve_func(stepping_kernel, dt)
    elif method == 'rk4':
        def stepping_kernel(_, rho, n, dt):
            return lindblad_kernel_rk4(rho, hmt, gamma_list, deco_list, hb, dt, n)

        evolve_func = get_stepping_evolve_func(stepping_kernel, dt)
    else:
        raise TypeError(f"Unsupported method {method}!")

    return evolve_func


def lindblad_kernel_euler(
        rho: np.ndarray,
        hmt: np.ndarray, gamma_list, deco_list,
        hb, dt, n):
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    k_sh = - 1j / hb
    for i in range(n):
        sh_part = hmt @ rho - rho @ hmt

        ln_part = np.sum([
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
        ], axis=0)

        rho += dt * (k_sh * sh_part + ln_part)
    return rho


def lindblad_kernel_rk4(
        rho: np.ndarray,
        hmt: np.ndarray, gamma_list, deco_list,
        hb, dt, n):
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    k_sh = - 1j / hb

    def dv(rho):
        sh_part = hmt @ rho - rho @ hmt

        ln_part = np.sum([
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
        ], axis=0)

        return k_sh * sh_part + ln_part

    for i in range(n):
        dt2 = dt / 2.0
        k1 = dv(rho)
        k2 = dv(rho + dt2 * k1)
        k3 = dv(rho + dt2 * k2)
        k4 = dv(rho + dt * k3)
        rho += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return rho


# kernels: differential stepping

def get_stepping_evolve_func(stepping_kernel, dt):
    def stepping_evolve_func(t, v, span):
        tiled_n = int(np.ceil(span / dt))
        tiled_dt = span / tiled_n
        return stepping_kernel(t, v, tiled_n, tiled_dt)

    return stepping_evolve_func


def get_differential_stepping_kernel(dv, method='euler'):
    if method is None:
        method = 'euler'
    else:
        method = str(method).lower()

    if method == 'euler':
        def stepping_kernel_func(t, v, n, dt):
            for i in range(n):
                v = one_step_euler(t, v, dt, dv)
            return v
    elif method == 'rk4':
        def stepping_kernel_func(t, v, n, dt):
            for i in range(n):
                v = one_step_rk4(t, v, dt, dv)
            return v
    else:
        raise TypeError(f"Unsupported method {method}")

    return stepping_kernel_func


def one_step_euler(t, v, dt, dv):
    return v + dt * dv(t, v)


def one_step_rk4(t, v, dt, dv):
    dt2 = dt / 2.0
    k1 = dv(t, v)
    k2 = dv(t + dt2, v + dt2 * k1)
    k3 = dv(t + dt2, v + dt2 * k2)
    k4 = dv(t + dt, v + dt * k3)
    return v + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# utils: wrap & unwrap

def unwrap(value: QTensor, operators, reduce=True, dtype=np.complex64):
    org_spaces = value.spaces
    if reduce:
        reduction = PrunedKetSpace.from_initial([value], [*structured_iter(operators)])
        value = reduction.reduce(value)
        operators = reduction.reduce(operators)
    else:
        reduction = None
        operators = structured_map(operators, lambda op: op.broadcast(value.spaces))

    flat_spaces, value = value.flatten()
    value = np.asarray(value, dtype=dtype)

    operators = structured_map(operators, lambda op: np.asarray(op.flattened_values, dtype=dtype))

    wrapping = org_spaces, flat_spaces, reduction
    return value, operators, wrapping


def wrap(value, wrapping) -> QTensor:
    org_spaces, flat_spaces, reduction = wrapping
    if reduction:
        value = QTensor.wrap(value, flat_spaces)
        value = reduction.inflate(value)
        value = value.transposed(org_spaces)
    else:
        value = QTensor.wrap(value, flat_spaces, org_spaces)
    return value


# utils: log

def get_logging_evolve_func(evolve_func, log_func=None, dlt=None, verbose=True):
    if dlt is None:
        return evolve_func

    def logging_evolve_func(t, v, span):
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

    return logging_evolve_func


def get_wrapping_log_func(log_func, wrapping):
    if log_func is None:
        return None

    def wrapping_log_func(t_log, psi_log):
        psi_log = wrap(psi_log, wrapping)
        return log_func(t_log, psi_log)

    return wrapping_log_func


# utils: rectify

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
