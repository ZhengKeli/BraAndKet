from ..kernel import AutoKernel

from .evolution import Evolution


def AutoEvolution(*,
        # model
        hb=1.0, hmt=None, deco=None,

        # initial
        time=0, value=None,

        # kernel configuration
        backend=None, method=None, dt=None,

        # evolution configuration
        check_dt=None, check_func=None, rectify=True, verbose=True,
):
    kernel = AutoKernel(
        hb=hb, hmt=hmt, deco=deco,
        time=time, value=value,
        backend=backend, method=method, dt=dt)
    return Evolution(kernel,
        check_dt=check_dt, check_func=check_func, rectify=rectify, verbose=verbose)
