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
    psi, hmt, wrapping = unwrap(psi, hmt, reduce=reduce)

    # compute
    evolve_func = get_schrodinger_evolve_func(hmt, hb, dt, method)

    evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    psi = evolve_func(t, psi, span)
    t = t + span

    # wrap
    psi = wrap(psi, wrapping)

    return t, psi


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
        t, psi, hmt, k_func, hb, span, dt,
        dlt=None, log_func=None,
        reduce=True, method=None, rectify=True,
        verbose=True):
    hmt_list = [hmt] if isinstance(hmt, QTensor) else list(hmt)
    hmt_count = len(hmt_list)

    # unwrap
    psi, hmt_list, wrapping = unwrap(psi, hmt_list, reduce=reduce)

    # prepare
    hmt_list = np.stack(hmt_list, -1)

    # compute
    def dv(t, psi):
        k_list = np.reshape(k_func(t), [hmt_count])
        hmt = np.sum(hmt_list * k_list, -1)
        return -1j / hb * (hmt @ psi)

    stepping_kernel = get_differential_stepping_kernel(dv, method)

    evolve_func = get_stepping_evolve_func(stepping_kernel, dt)

    evolve_func = get_rectifying_psi_evolve_func(evolve_func, rectify)

    log_func = get_wrapping_log_func(log_func, wrapping)

    evolve_func = get_logging_evolve_func(evolve_func, log_func, dlt, verbose)

    psi = evolve_func(t, psi, span)
    t = t + span

    # wrap
    psi = wrap(psi, wrapping)

    return t, psi


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
            def stepping_kernel(_, psi, n, dt):
                return schrodinger_kernel_euler(psi, hmt, hb, dt, n)
        elif method == 'rk4':
            def stepping_kernel(_, psi, n, dt):
                return schrodinger_kernel_rk4(psi, hmt, hb, dt, n)
        else:
            raise TypeError(f"Unsupported method {method}!")

        evolve_func = get_stepping_evolve_func(stepping_kernel, dt)
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
    org_dims = value.dims
    if reduce:
        reduction = ReducedHSpace.from_initial([value], [*structured_iter(operators)])
        value = reduction.reduce(value)
        operators = structured_map(operators, reduction.reduce)
    else:
        reduction = None
        operators = structured_map(operators, lambda op: op.broadcast(value.dims))

    flat_dims, value = value.flatten()
    value = np.asarray(value, dtype=dtype)

    operators = structured_map(operators, lambda op: op.flattened_values)
    operators = structured_map(operators, lambda op: np.asarray(op, dtype=dtype))

    wrapping = org_dims, flat_dims, reduction
    return value, operators, wrapping


def wrap(value, wrapping) -> QTensor:
    org_dims, flat_dims, reduction = wrapping
    if reduction:
        value = QTensor.wrap(value, flat_dims)
        value = reduction.inflate(value)
        value = value.transposed(org_dims)
    else:
        value = QTensor.wrap(value, flat_dims, org_dims)
    return value


def structured_iter(structure):
    if isinstance(structure, (list, tuple)):
        for item in structure:
            for sub_item in structured_iter(item):
                yield sub_item
    else:
        yield structure


def structured_map(structure, map_func):
    if isinstance(structure, list):
        return [structured_map(item, map_func) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(structured_map(item, map_func) for item in structure)
    else:
        return map_func(structure)


# utils: log

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
