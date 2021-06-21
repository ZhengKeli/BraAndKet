from ..kernel import AutoKernel

from .evolution import Evolution


def AutoEvolution(*,
        hb=1.0, hmt=None, deco=None,
        time=0, value=None,
        backend=None, method=None,
        check_dt=None, check_func=None, rectify=True, verbose=True,
        **kwargs
):
    kernel = AutoKernel(
        hb=hb, hmt=hmt, deco=deco,
        time=time, value=value,
        backend=backend, method=method,
        **kwargs)
    return Evolution(kernel,
        check_dt=check_dt, check_func=check_func, rectify=rectify, verbose=verbose)
