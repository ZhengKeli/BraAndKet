import numpy as np

from ..space import BraSpace
from ..tensor import QTensor, zero, NumpyQTensor, SparseQTensor
from ..utils import structured_iter


def merge_optional_iter(iter1, iter2):
    if iter1 is None:
        if iter2 is None:
            return None
        else:
            return iter2
    else:
        if iter2 is None:
            return iter1
        else:
            return iter((*iter1, *iter2))


def dirty_tensor(tensor: QTensor):
    if isinstance(tensor, NumpyQTensor):
        return NumpyQTensor(tensor.spaces, tensor.values != 0)
    elif isinstance(tensor, SparseQTensor):
        return SparseQTensor(tensor.spaces, ((coordinate, 1) for coordinate, value in tensor.values))
    else:
        assert False


def dirty_rho_to_psi(tensor: QTensor):
    if isinstance(tensor, NumpyQTensor):
        bra_axes = []
        new_spaces = []
        for axis, space in enumerate(tensor.spaces):
            if isinstance(space, BraSpace):
                bra_axes.append(axis)
            else:
                new_spaces.append(space)
        new_values = np.any(tensor.values, bra_axes)
        return NumpyQTensor(new_spaces, new_values)
    elif isinstance(tensor, SparseQTensor):
        new_spaces = []
        new_axes = []
        for axis, space in enumerate(tensor.spaces):
            if not isinstance(space, BraSpace):
                new_spaces.append(space)
                new_axes.append(axis)
        new_values = ((tuple(coordinate[axis] for axis in new_axes), 1) for coordinate, value in tensor.values)
        return SparseQTensor(new_spaces, new_values)
    else:
        assert False


def extract_dirty_eigenstates(initial, operators):
    all_psi = zero
    for tensor in structured_iter(initial):
        if tensor.is_psi:
            psi = dirty_tensor(tensor)
        else:
            tensor_ct = tensor.ct
            if tensor_ct.is_psi:
                psi = dirty_tensor(tensor_ct)
            elif tensor.is_rho:
                psi = dirty_rho_to_psi(tensor)
            else:
                raise TypeError("the tensors in parameter initial should be either vector or density matrix.")
        all_psi += psi
    all_psi = dirty_tensor(all_psi)
    all_op = zero
    for op in structured_iter(operators):
        all_op += dirty_tensor(op)
    all_op = dirty_tensor(all_op)
    while True:
        new_psi = all_op @ all_psi
        new_psi += all_psi
        new_psi = dirty_tensor(new_psi)
        if new_psi == all_psi:
            break
        all_psi = new_psi
    if isinstance(all_psi, NumpyQTensor):
        coordinates = np.transpose(np.where(all_psi.values))
    elif isinstance(all_psi, SparseQTensor):
        coordinates, _ = zip(*all_psi.values)
    else:
        assert False
    eigenstates = tuple(
        SparseQTensor(all_psi.spaces, [(coordinate, True)])
        for coordinate in coordinates
    )
    return eigenstates
