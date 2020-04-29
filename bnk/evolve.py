import numpy as np

from bnk import QTensor


# accelerated evolve functions

def evolve_schrodinger(psi, hmt, hb, dt, mt):
    n = int(np.ceil(mt / dt))
    dt = mt / n
    
    hmt = hmt.broadcast(psi.dims)
    
    # unwrap
    _, hmt = hmt.flatten()
    psi_dims, psi = psi.flatten()
    hmt = np.asarray(hmt, dtype=np.complex64)
    psi = np.asarray(psi, dtype=np.complex64)
    
    kt = (dt / 1j / hb)
    for i in range(n):
        psi += kt * (hmt @ psi)
    
    # wrap
    psi = psi.reshape([dim.n for group in psi_dims for dim in group])
    psi = QTensor([dim for group in psi_dims for dim in group], np.copy(psi))
    
    return psi


def evolve_lindblad(rho, hmt, deco, hb, dt, mt):
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


# evolve functions with logs

def evolve_with_logs(t0, v0, span, dlt, evolve_func, log_func=None, verbose=True):
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


def evolve_schrodinger_with_logs(t0, psi0, hmt, hb, span, dt, dlt, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, psi, sp):
        psi = evolve_schrodinger(psi, hmt, hb, dt, sp)
        if rectify:
            psi /= np.sqrt(np.sum(np.conj(psi.values) * psi.values))
        psi = psi.transposed(psi0.dims)
        return psi
    
    return evolve_with_logs(t0, psi0, span, dlt, evolve_func, log_func, verbose)


def evolve_lindblad_with_logs(t0, rho0, hmt, deco, hb, span, dt, dlt, log_func=None, rectify=True, verbose=True):
    def evolve_func(t, rho, sp):
        rho = evolve_lindblad(rho, hmt, deco, hb, dt, sp)
        if rectify:
            rho /= rho.trace().values
        rho = rho.transposed(rho0.dims)
        return rho
    
    return evolve_with_logs(t0, rho0, span, dlt, evolve_func, log_func, verbose)
