import numpy as np

from bnk import QTensor


def evolve_schrodinger(psi, hmt, hb, span, dt, rectify=True):
    n = int(np.ceil(span / dt))
    dt = span / n
    
    org_psi_dims = psi.dims
    hmt_dims, hmt_matrix = hmt.broadcast(psi.dims).flatten()
    psi_dims, psi_matrix = psi.flatten()
    
    hmt_matrix = np.asarray(hmt_matrix, dtype=np.complex64)
    psi_matrix = np.asarray(psi_matrix, dtype=np.complex64)
    
    for i in range(n):
        psi_matrix += np.dot(hmt_matrix, psi_matrix) * (dt / 1j / hb)
    
    if rectify:
        psi_matrix /= np.sqrt(np.sum(np.conj(psi_matrix) * psi_matrix))
    
    psi_matrix = psi_matrix.reshape([dim.n for group in psi_dims for dim in group])
    psi = QTensor([dim for group in psi_dims for dim in group], psi_matrix)
    
    return psi.transposed(org_psi_dims)


def evolve_schrodinger_with_logs(psi0, hmt, hb, mt, dt, dlt, rectify=True, logger=None, verbose=True):
    if logger is None:
        logger = lambda ps: ps.values
    
    t = 0.0
    psi = psi0
    
    lt = t
    logs_t = [lt]
    logs_v = [logger(psi)]
    
    while t < mt:
        sp = np.maximum(np.minimum(lt + dlt, mt) - t, 0.0)
        psi = evolve_schrodinger(psi, hmt, hb, sp, dt, rectify=rectify)
        
        t += sp
        lt = t
        logs_t.append(t)
        logs_v.append(logger(psi))
        if verbose:
            print(f"\rcomputing...{t / mt:.2%}", end='')
    print()
    
    logs_t = np.asarray(logs_t)
    try:
        logs_v = np.asarray(logs_v)
    except RuntimeError:
        pass
    
    return psi, logs_t, logs_v


def evolve_lindblad(rho, hmt, deco, span, dt, hb):
    pass
