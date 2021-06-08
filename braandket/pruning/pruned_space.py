from typing import Iterable

from .utils import extract_dirty_eigenstates
from ..space import KetSpace
from ..tensor import QTensor
from ..utils import structured_map, sum


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
        eigenstates = extract_dirty_eigenstates(initial, operators)
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
