from typing import TypeVar

import numpy as np

from .space import PrunedKetSpace
from ..model import QModel


class PrunedQModel(QModel):
    AnyQModel = TypeVar('AnyQModel', bound=QModel)

    def __init__(self, org_model: AnyQModel, initial):
        self.org = org_model

        operators = [org_model.hmt]
        if org_model.deco is not None:
            operators.append(org_model.deco)
        self.space = PrunedKetSpace.from_initial(initial, operators)

        hmt = self.reduce(org_model.hmt)
        if org_model.deco is not None:
            gamma = org_model.gamma
            deco = self.reduce(org_model.deco)
        else:
            gamma = None
            deco = None
        super().__init__(hmt, gamma, deco, org_model.hb)

    @property
    def n(self):
        return self.space.n

    def eigenstate(self, index, dtype=np.float32):
        return self.space.eigenstate(index, dtype=dtype)

    def reduce(self, tensor):
        return self.space.prune(tensor)

    def inflate(self, tensor):
        return self.space.restore(tensor)
