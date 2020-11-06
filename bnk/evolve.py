import numpy as np
from tqdm import tqdm

from bnk.tensor import QTensor


# accelerated evolve functions

def _raw_schrodinger_evolve(psi, hmt, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n

    hmt = hmt.broadcast(psi.dims)

    # unwrap
    _, hmt = hmt.flatten()
    psi_dims, psi = psi.flatten()
    hmt = np.asarray(hmt, dtype=np.complex64)
    psi = np.asarray(psi, dtype=np.complex64)

    # compute
    kt = (dt / 1j / hb)
    for i in range(n):
        psi += kt * (hmt @ psi)

    # wrap
    psi = psi.reshape([dim.n for group in psi_dims for dim in group])
    psi = QTensor([dim for group in psi_dims for dim in group], np.copy(psi))

    return psi


def _raw_lindblad_evolve(rho, hmt, deco, gamma, hb, dt, mt):
    deco_list = [deco] if isinstance(deco, QTensor) else list(deco)
    gamma_list = np.broadcast_to(gamma, [len(deco_list)])

    n = int(np.ceil(mt / dt))
    dt = mt / n

    # broadcast
    all_dims = {
        *rho.dims,
        *hmt.dims,
        *(dim for deco in deco_list for dim in deco.dims)
    }
    rho = rho.broadcast(all_dims)
    hmt = hmt.broadcast(all_dims)
    deco_list = [deco.broadcast(all_dims) for deco in deco_list]

    # unwrap
    rho_dims, rho = rho.flatten()
    rho = np.asarray(rho, dtype=np.complex64)

    hmt = hmt.flatten()[1]
    hmt = np.asarray(hmt, dtype=np.complex64)

    deco_list = [deco.flatten()[1] for deco in deco_list]
    deco_list = [np.asarray(deco, dtype=np.complex64) for deco in deco_list]
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    # compute
    k_sh = - 1j / hb
    for i in range(n):
        sh_part = hmt @ rho - rho @ hmt

        ln_part = sum(
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
        )

        rho += dt * (k_sh * sh_part + ln_part)

    # wrap
    rho = rho.reshape([dim.n for group in rho_dims for dim in group])
    rho = QTensor([dim for group in rho_dims for dim in group], np.copy(rho))

    return rho


def _raw_dynamic_schrodinger_evolve(psi, hmt_list, k_list_func, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n

    hmt_list = list(hmt_list)
    for i in range(len(hmt_list)):
        hmt = hmt_list[i]
        hmt = hmt.broadcast(psi.dims)
        _, hmt = hmt.flatten()
        hmt = np.asarray(hmt, dtype=np.complex64)
        hmt_list[i] = hmt
    hmt_list = np.stack(hmt_list, -1)

    psi_dims, psi = psi.flatten()
    psi = np.asarray(psi, dtype=np.complex64)

    # compute
    kt = (dt / 1j / hb)
    for i in range(n):
        k_list = np.asarray(k_list_func((i + 0.5) * dt))
        hmt = np.sum(hmt_list * k_list, -1)
        psi += kt * (hmt @ psi)

    # wrap
    psi = psi.reshape([dim.n for group in psi_dims for dim in group])
    psi = QTensor([dim for group in psi_dims for dim in group], np.copy(psi))

    return psi


def _raw_dynamic_lindblad_evolve(rho, hmt_list, k_list_func, deco_list, gamma_list_func, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n

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
    rho_dims, rho = rho.flatten()
    rho = np.asarray(rho, dtype=np.complex64)

    hmt_list = [hmt.flatten()[1] for hmt in hmt_list]
    hmt_list = [np.asarray(hmt, dtype=np.complex64) for hmt in hmt_list]

    deco_list = [deco.flatten()[1] for deco in deco_list]
    deco_list = [np.asarray(deco, dtype=np.complex64) for deco in deco_list]
    deco_ct_list = [np.conj(np.transpose(deco)) for deco in deco_list]
    deco_ct_deco_list = [deco_ct @ deco for deco, deco_ct in zip(deco_list, deco_ct_list)]

    # compute
    hmt_list = np.stack(hmt_list, -1)
    k_sh = - 1j / hb
    for i in range(n):
        t = (i + 0.5) * dt

        k_list = k_list_func(t)
        k_list = np.asarray(k_list)
        hmt = np.sum(hmt_list * k_list, -1)
        sh_part = hmt @ rho - rho @ hmt

        gamma_list = gamma_list_func(t)
        ln_part = sum(
            gamma * (deco @ rho @ deco_ct - 0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco))
            for gamma, deco, deco_ct, deco_ct_deco
            in zip(gamma_list, deco_list, deco_ct_list, deco_ct_deco_list)
        )

        rho += dt * (k_sh * sh_part + ln_part)

    # wrap
    rho = rho.reshape([dim.n for group in rho_dims for dim in group])
    rho = QTensor([dim for group in rho_dims for dim in group], np.copy(rho))

    return rho


# evolve functions with logs

def evolve(t0, v0, span, evolve_func,
           dlt=None, log_func=None, verbose=True):
    dlt = (span / 100) if dlt is None else dlt

    mt = t0 + span
    progress = None

    t = t0
    v = v0

    if log_func:
        log_func(t, v)
    if verbose:
        progress = tqdm(total=span)

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

    return t, v


def schrodinger_evolve(t0, psi0, hmt, hb, span, dt,
                       dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, psi, sp):
        psi = _raw_schrodinger_evolve(psi, hmt, hb, dt, sp)
        if rectify:
            psi /= np.sqrt(np.sum(np.conj(psi.values) * psi.values))
        psi = psi.transposed(psi0.dims)
        return psi

    return evolve(t0, psi0, span, evolve_func, dlt, log_func, verbose)


def lindblad_evolve(t0, rho0, hmt, deco, gamma, hb, span, dt,
                    dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, rho, sp):
        rho = _raw_lindblad_evolve(rho, hmt, deco, gamma, hb, dt, sp)
        if rectify:
            rho /= rho.trace().values
        rho = rho.transposed(rho0.dims)
        return rho

    return evolve(t0, rho0, span, evolve_func, dlt, log_func, verbose)


def dynamic_schrodinger_evolve(t0, psi0, hmt_list, k_list_func, hb, span, dt,
                               dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, psi, sp):
        psi = _raw_dynamic_schrodinger_evolve(psi, hmt_list, lambda ti: k_list_func(ti + t), hb, dt, sp)
        if rectify:
            psi /= np.sqrt(np.sum(np.conj(psi.values) * psi.values))
        psi = psi.transposed(psi0.dims)
        return psi

    return evolve(t0, psi0, span, evolve_func, dlt, log_func, verbose)


def dynamic_lindblad_evolve(t0, rho0, hmt_list, k_list_func, deco_list, gamma_list_func, hb, span, dt,
                            dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, rho, sp):
        rho = _raw_dynamic_lindblad_evolve(
            rho,
            hmt_list, lambda ti: k_list_func(ti + t),
            deco_list, lambda ti: gamma_list_func(ti + t),
            hb, dt, sp)
        if rectify:
            rho /= np.real(rho.trace().values)
        rho = rho.transposed(rho0.dims)
        return rho

    return evolve(t0, rho0, span, evolve_func, dlt, log_func, verbose)
