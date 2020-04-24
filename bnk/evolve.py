import numpy as np

from bnk import QTensor


def evolve_schrodinger(psi, hmt, hb, mt, dt):
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


def evolve_schrodinger_with_logs(psi0, hmt, hb, mt, dt, dlt, t0=0.0, rectify=True, logger=None, verbose=True):
    if logger is None:
        logger = lambda ps: ps.values
    
    t = t0
    psi = psi0
    
    lt = t
    logs_t = [lt]
    logs_v = [logger(psi)]
    
    while t < mt:
        sp = np.maximum(np.minimum(lt + dlt, mt) - t, 0.0)
        psi = evolve_schrodinger(psi, hmt, hb, sp, dt)
        
        if rectify:
            psi /= np.sqrt(np.sum(np.conj(psi.values) * psi.values))
        
        psi = psi.transposed(psi0.dims)
        
        t += sp
        lt = t
        logs_t.append(t)
        logs_v.append(logger(psi))
        
        if verbose:
            print(f"\rcomputing...{t / mt:.2%}", end='')
    
    if verbose:
        print(f"\rcomputing...{1.0:.2%}")
    
    logs_t = np.asarray(logs_t)
    try:
        logs_v = np.asarray(logs_v)
    except RuntimeError:
        pass
    
    return psi, logs_t, logs_v


def evolve_lindblad(rho, hmt, deco, hb, mt, dt):
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


def evolve_lindblad_with_logs(rho0, hmt, deco, hb, mt, dt, dlt, t0=0.0, rectify=True, logger=None, verbose=True):
    if logger is None:
        logger = lambda ro: ro.values
    
    t = t0
    rho = rho0
    
    lt = t
    logs_t = [lt]
    logs_v = [logger(rho)]
    
    while t < mt:
        sp = np.maximum(np.minimum(lt + dlt, mt) - t, 0.0)
        rho = evolve_lindblad(rho, hmt, deco, hb, sp, dt)
        
        if rectify:
            rho /= rho.trace().values
        
        rho = rho.transposed(rho0.dims)
        
        t += sp
        lt = t
        logs_t.append(t)
        logs_v.append(logger(rho))
        
        if verbose:
            print(f"\rcomputing...{t / mt:.2%}", end='')
    
    if verbose:
        print(f"\rcomputing...{1.0:.2%}")
    
    logs_t = np.asarray(logs_t)
    try:
        logs_v = np.asarray(logs_v)
    except RuntimeError:
        pass
    
    return rho, logs_t, logs_v
