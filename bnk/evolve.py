import numpy as np

from bnk import QTensor


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


def _raw_lindblad_evolve(rho, hmt, deco, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n
    
    all_dims = [*rho.dims, *hmt.dims, *deco.dims]
    hmt = hmt.broadcast(rho.dims)
    deco = deco.broadcast(all_dims)
    rho = rho.broadcast(all_dims)
    
    # unwrap
    _, hmt = hmt.flatten()
    _, deco = deco.flatten()
    rho_dims, rho = rho.flatten()
    hmt = np.asarray(hmt, dtype=np.complex64)
    deco = np.asarray(deco, dtype=np.complex64)
    rho = np.asarray(rho, dtype=np.complex64)
    
    # compute
    kt = (dt / 1j / hb)
    deco_ct = np.conj(np.transpose(deco))
    deco_ct_deco = np.dot(deco_ct, deco)
    for i in range(n):
        rho += kt * (
            hmt @ rho - rho @ hmt +
            1j * (
                deco @ rho @ deco_ct -
                0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco)
            )
        )
    
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


def _raw_dynamic_lindblad_evolve(rho, hmt_list, deco_list, k_list_func, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n
    
    all_dims = []
    all_dims.extend(rho.dims)
    for hmt in hmt_list:
        all_dims.extend(hmt.dims)
    for deco in deco_list:
        all_dims.extend(deco.dims)
    
    rho = rho.broadcast(all_dims)
    
    hmt_list = list(hmt_list)
    for i in range(len(hmt_list)):
        hmt = hmt_list[i]
        hmt = hmt.broadcast(rho.dims)
        _, hmt = hmt.flatten()
        hmt = np.asarray(hmt, dtype=np.complex64)
        hmt_list[i] = hmt
    hmt_list = np.stack(hmt_list, -1)
    
    deco_list = list(deco_list)
    deco_ct_deco_list = []
    for i in range(len(deco_list)):
        deco = deco_list[i]
        deco = deco.broadcast(rho.dims)
        _, deco = deco.flatten()
        deco = np.asarray(deco, dtype=np.complex64)
        deco_list[i] = deco
        
        deco_ct = np.conj(np.transpose(deco))
        deco_ct_deco = deco_ct @ deco
        deco_ct_deco_list.append(deco_ct_deco)
    deco_list = np.stack(deco_list, -1)
    deco_ct_deco_list = np.stack(deco_ct_deco_list, -1)
    
    rho_dims, rho = rho.flatten()
    rho = np.asarray(rho, dtype=np.complex64)
    
    # compute
    kt = (dt / 1j / hb)
    for i in range(n):
        hmt_k_list, deco_k_list = k_list_func((i + 0.5) * dt)
        hmt_k_list = np.asarray(hmt_k_list)
        deco_k_list = np.asarray(deco_k_list)
        
        hmt = np.sum(hmt_list * hmt_k_list, -1)
        deco = np.sum(deco_list * deco_k_list, -1)
        deco_ct = np.conj(np.transpose(deco))
        deco_ct_deco = np.sum(deco_ct_deco_list * (deco_k_list * deco_k_list), -1)
        
        rho += kt * (
            hmt @ rho - rho @ hmt +
            1j * (
                deco @ rho @ deco_ct -
                0.5 * (deco_ct_deco @ rho + rho @ deco_ct_deco)
            )
        )
    
    # wrap
    rho = rho.reshape([dim.n for group in rho_dims for dim in group])
    rho = QTensor([dim for group in rho_dims for dim in group], np.copy(rho))
    
    return rho


# evolve functions with logs

def evolve(t0, v0, span, evolve_func,
           dlt=None, log_func=None, verbose=True):
    dlt = (span / 100) if dlt is None else dlt

    mt = t0 + span
    
    t = t0
    v = v0
    
    if log_func is not None:
        log_func(t, v)
    if verbose:
        print(f"\revolving...{(t - t0) / span:.2%}", end='')
    
    while True:
        rt = mt - t
        if rt < dlt:
            break
        
        v = evolve_func(t, v, dlt)
        t += dlt
        
        if log_func is not None:
            log_func(t, v)
        if verbose:
            print(f"\revolving...{(t - t0) / span:.2%}", end='')
    
    if rt > 0:
        v = evolve_func(t, v, rt)
        t = mt
    
    if log_func is not None:
        log_func(t, v)
    if verbose:
        print(f"\revolving...{(t - t0) / span:.2%}")
    
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


def lindblad_evolve(t0, rho0, hmt, deco, hb, span, dt,
                    dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, rho, sp):
        rho = _raw_lindblad_evolve(rho, hmt, deco, hb, dt, sp)
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


def dynamic_lindblad_evolve(t0, rho0, hmt_list, deco_list, k_list_func, hb, span, dt,
                            dlt=None, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, rho, sp):
        rho = _raw_dynamic_lindblad_evolve(rho, hmt_list, deco_list, lambda ti: k_list_func(ti + t), hb, dt, sp)
        if rectify:
            rho /= np.real(rho.trace().values)
        rho = rho.transposed(rho0.dims)
        return rho

    return evolve(t0, rho0, span, evolve_func, dlt, log_func, verbose)
