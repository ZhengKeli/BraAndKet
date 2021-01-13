from typing import Iterable

import numpy as np

from bnk import KetSpace
from bnk.tensor import QTensor, zero, one
from bnk.utils import structured_iter, structured_map


class ReducedKetSpace(KetSpace):
    def __init__(self, org_eigenstates, name=None, key=None):
        super().__init__(len(org_eigenstates), name, key)

        org_eigenstates = tuple(org_eigenstates)

        transform = zero
        for i, org_eigenstate in enumerate(org_eigenstates):
            transform += self.eigenstate(i) @ org_eigenstate.ct

        self.org_eigenstates = org_eigenstates
        self.transform = transform

    @staticmethod
    def from_initial(initial: Iterable[QTensor], operators: Iterable[QTensor], name=None, key=None):
        all_psi = zero
        for tensor in structured_iter(initial):
            if all(dim.is_ket for dim in tensor.dims):
                psi = to_bool(tensor)
            elif all(dim.is_ket for dim in tensor.dims):
                psi = to_bool(tensor.ct)
            else:
                rho = to_bool(tensor)
                (ket_dims, bra_dims), flat_values = rho.flatten()
                if not all(ket_dim == bra_dim.ct for ket_dim, bra_dim in zip(ket_dims, bra_dims)):
                    raise TypeError("the tensors in parameter initial should be either vector or density matrix.")
                psi = QTensor.wrap(np.any(flat_values, axis=-1), (ket_dims,))
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

        indices = np.transpose(np.where(all_psi.values))
        eigenstates = []
        for index in indices:
            eigenstate = one
            for i, dim in zip(index, all_psi.dims):
                eigenstate @= dim.eigenstate(i)
            eigenstates.append(eigenstate)

        return ReducedKetSpace(eigenstates, name, key)

    def org_eigenstate(self, index):
        return self.org_eigenstates[index]

    def _reduce_one(self, tensor: QTensor):
        has_ket = False
        has_bra = False
        for dim in tensor.dims:
            if dim.is_ket:
                has_ket = True
            elif dim.is_bra:
                has_bra = True
        if has_ket:
            tensor = self.transform @ tensor
        if has_bra:
            tensor = tensor @ self.transform.ct
        return tensor

    def reduce(self, tensor):
        return structured_map(tensor, self._reduce_one)

    def _inflate_one(self, tensor: QTensor):
        has_ket = False
        has_bra = False
        for dim in tensor.dims:
            if dim.is_ket:
                has_ket = True
            elif dim.is_bra:
                has_bra = True
        if has_ket:
            tensor = self.transform.ct @ tensor
        if has_bra:
            tensor = tensor @ self.transform
        return tensor

    def inflate(self, tensor):
        return structured_map(tensor, self._inflate_one)


# utils

def to_bool(t: QTensor):
    return QTensor(t.dims, t.values != 0)
