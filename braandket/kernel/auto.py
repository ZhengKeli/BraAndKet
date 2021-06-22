from .flattening import SchrodingerRhoPade, SchrodingerPsiPade, LindbladEuler, LindbladRk4
from ..tensor import QTensor


def AutoKernel(*,
        # model
        hb=1.0, hmt=None, deco=None,

        # initial
        time=0, value=None,

        # kernel configurations
        backend=None, method=None, dt=None,
):
    static_hmt, dynamic_hmt = dynamic_static_op_list(hmt)
    static_deco, dynamic_deco = dynamic_static_op_list(deco)
    model = hb, static_hmt, static_deco, dynamic_hmt, dynamic_deco

    if not isinstance(value, QTensor):
        raise TypeError("psi_or_rho is expected to be a QTensor, "
                        f"got {value}")

    backend = 'numpy' if backend is None else backend

    if len(static_deco) + len(dynamic_deco) == 0:  # schrodinger
        if value.is_psi:
            is_rho = False
        elif value.is_rho:
            is_rho = True
        else:
            raise TypeError(
                "psi_or_rho is expected to be a pure state vector (psi) or a density matrix (rho), "
                f"got {value}")

        if len(dynamic_hmt) == 0:  # static schrodinger
            if backend == 'numpy':
                method = 'pade' if method is None else method
                if method == 'pade':  # pade approximation
                    if is_rho:
                        kernel = SchrodingerRhoPade(model, time, value)
                    else:
                        kernel = SchrodingerPsiPade(model, time, value)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:  # dynamic schrodinger
            raise NotImplementedError()
    else:  # lindblad
        if len(dynamic_hmt) + len(dynamic_deco) == 0:  # static lindblad
            if backend == 'numpy':
                method = 'euler' if method is None else method
                if method == 'euler':
                    kernel = LindbladEuler(model, time, value, dt=dt)
                elif method == 'rk4':
                    kernel = LindbladRk4(model, time, value, dt=dt)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:  # dynamic lindblad
            raise NotImplementedError()

    return kernel


# utils

def dynamic_static_op_list(op_list):
    op_list = formal_op_list(op_list)

    static_op_list = []
    dynamic_op_list = []
    for k, op in op_list:
        if callable(k):
            dynamic_op_list.append((k, op))
        else:
            static_op_list.append((k, op))

    return tuple(static_op_list), tuple(dynamic_op_list)


def formal_op_list(op_list):
    if op_list is None:
        return tuple()

    if isinstance(op_list, QTensor):
        op = op_list
        return (1.0, op),

    op_list = tuple(op_list)
    try:
        op_list = tuple(formal_op_item(op_item) for op_item in op_list)
    except TypeError:
        op_item = op_list
        op_item = formal_op_item(op_item)
        op_list = op_item,

    return op_list


def formal_op_item(op_item):
    if isinstance(op_item, QTensor):
        return 1.0, op_item

    op_item = tuple(op_item)
    if not len(op_item) == 2:
        raise TypeError()

    k, op = op_item
    if isinstance(k, QTensor) and not k.is_scalar:
        raise TypeError()
    if not isinstance(op, QTensor):
        raise TypeError()
    return k, op
