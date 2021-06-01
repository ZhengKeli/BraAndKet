from .utils import merge_optional_iter, extract_dirty_eigenstates
from ..model import QModel, QComponent
from ..space import KetSpace
from ..tensor import QTensor
from ..utils import structured_map, sum


class PrunedQComponent(QComponent):
    def __init__(self, component: QComponent, initial=None, operators=None):
        initial = merge_optional_iter(initial, component.initial)
        if initial is None:
            raise ValueError("Can not prune this component without initial.")

        operators = merge_optional_iter(operators, component.operators)
        if operators is None:
            raise ValueError("Can not prune this component without operators.")

        org_eigenstates = extract_dirty_eigenstates(initial, operators)
        org_eigenstates = tuple(org_eigenstates)

        space = KetSpace(len(org_eigenstates), name="pruned")
        super().__init__(space)

        op_prune = sum(
            self.eigenstate(i) @ org_eigenstate.ct
            for i, org_eigenstate in enumerate(org_eigenstates)
        )

        self._org_eigenstates = org_eigenstates
        self._op_prune = op_prune
        self._op_restore = op_prune.ct

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


class PrunedQModel(PrunedQComponent, QModel):
    def __init__(self, model: QModel, initial=None, operators=None):
        initial = merge_optional_iter(initial, model.initial)
        operators = merge_optional_iter(operators, model.operators)

        hmt = structured_map(model.hmt, self.prune)
        gamma = model.gamma
        deco = structured_map(model.deco, self.prune)
        hb = model.hb

        super().__init__(model, initial, operators)
        QModel.__init__(self, self.children, hmt, gamma, deco, hb, initial, operators)
