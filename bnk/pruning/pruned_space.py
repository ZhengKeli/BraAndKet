from typing import Iterable

import numpy as np

from ..space import KetSpace, BraSpace
from ..tensor import QTensor, zero, NumpyQTensor, SparseQTensor
from ..utils import structured_iter, structured_map, sum


class PrunedKetSpace(KetSpace):
    def __init__(self, org_eigenstates, name=None):
        super().__init__(len(org_eigenstates), name)

        org_eigenstates = tuple(org_eigenstates)

        transform = sum(
            self.eigenstate(i) @ org_eigenstate.ct
            for i, org_eigenstate in enumerate(org_eigenstates)
        )

        self._org_eigenstates = org_eigenstates
        self._op_prune = transform
        self._op_restore = transform.ct

    @staticmethod
    def from_initial(initial: Iterable[QTensor], operators: Iterable[QTensor], name=None):
        all_psi = zero
        for tensor in structured_iter(initial):
            if tensor.is_psi:
                psi = to_bool(tensor)
            else:
                tensor_ct = tensor.ct
                if tensor_ct.is_psi:
                    psi = to_bool(tensor_ct)
                elif tensor.is_rho:
                    psi = rho_to_psi(tensor)
                else:
                    raise TypeError("the tensors in parameter initial should be either vector or density matrix.")
            all_psi += psi
        all_psi = to_bool(all_psi)

        all_op = zero
        for op in structured_iter(operators):
            all_op += to_bool(op)
        all_op = to_bool(all_op)

        while True:
            new_psi = all_op @ all_psi
            new_psi += all_psi
            new_psi = to_bool(new_psi)
            if new_psi == all_psi:
                break
            all_psi = new_psi

        if isinstance(all_psi, NumpyQTensor):
            coordinates = np.transpose(np.where(all_psi.values))
        elif isinstance(all_psi, SparseQTensor):
            coordinates, _ = zip(*all_psi.values)
        else:
            assert False

        eigenstates = [
            SparseQTensor(all_psi.spaces, [(coordinate, True)])
            for coordinate in coordinates
        ]
        return PrunedKetSpace(eigenstates, name)

    @property
    def org_eigenstates(self):
        return self._org_eigenstates

    def org_eigenstate(self, index):
        return self.org_eigenstates[index]

    def _prune_one(self, tensor: QTensor):
        has_ket = False
        has_bra = False
        for space in tensor.spaces:
            if space.is_ket:
                has_ket = True
                if has_bra:
                    break
            elif space.is_bra:
                has_bra = True
                if has_ket:
                    break
        if has_ket:
            tensor = self._op_prune @ tensor
        if has_bra:
            tensor = tensor @ self._op_restore
        return tensor

    def prune(self, tensor):
        return structured_map(tensor, self._prune_one)

    def _restore_one(self, tensor: QTensor):
        has_ket = False
        has_bra = False
        for space in tensor.spaces:
            if space.is_ket:
                has_ket = True
            elif space.is_bra:
                has_bra = True
        if has_ket:
            tensor = self._op_restore @ tensor
        if has_bra:
            tensor = tensor @ self._op_prune
        return tensor

    def restore(self, tensor):
        return structured_map(tensor, self._restore_one)


# utils

def to_bool(t: QTensor):
    if isinstance(t, NumpyQTensor):
        return NumpyQTensor(t.spaces, t.values != 0)
    elif isinstance(t, SparseQTensor):
        return SparseQTensor(t.spaces, ((coordinate, 1) for coordinate, value in t.values))
    else:
        assert False


def rho_to_psi(t: QTensor):
    if isinstance(t, NumpyQTensor):
        bra_axes = []
        new_spaces = []
        for axis, space in enumerate(t.spaces):
            if isinstance(space, BraSpace):
                bra_axes.append(axis)
            else:
                new_spaces.append(space)
        new_values = np.any(t.values, bra_axes)
        return NumpyQTensor(new_spaces, new_values)
    elif isinstance(t, SparseQTensor):
        new_spaces = []
        new_axes = []
        for axis, space in enumerate(t.spaces):
            if not isinstance(space, BraSpace):
                new_spaces.append(space)
                new_axes.append(axis)
        new_values = ((tuple(coordinate[axis] for axis in new_axes), 1) for coordinate, value in t.values)
        return SparseQTensor(new_spaces, new_values)
    else:
        assert False
