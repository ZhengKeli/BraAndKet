from typing import TypeVar

import numpy as np

from .tensor import QTensor
from .evolve import lindblad_evolve, schrodinger_evolve
from .reduce import ReducedKetSpace


class QModel:
    def __init__(self, hmt, gamma=None, deco=None, hb=1):
        self.hb = hb

        self.hmt = hmt
        self.gamma = gamma
        self.deco = deco

    def evolve(self, t, psi_or_rho: QTensor, span, dt=None, *args, **kwargs):
        if self.deco is None or self.gamma is None or np.all(np.asarray(self.gamma) == 0):
            return schrodinger_evolve(
                t, psi_or_rho,
                self.hmt, self.hb,
                span, dt,
                *args, **kwargs)
        else:
            rho = psi_or_rho.as_rho()
            return lindblad_evolve(
                t, rho,
                self.hmt, self.deco, self.gamma, self.hb,
                span, dt,
                *args, **kwargs)


class ReducedQModel(QModel):
    AnyQModel = TypeVar('AnyQModel', bound=QModel)

    def __init__(self, org_model: AnyQModel, initial):
        self.org = org_model

        operators = [org_model.hmt]
        if org_model.deco is not None:
            operators.append(org_model.deco)
        self.space = ReducedKetSpace.from_initial(initial, operators)

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
        return self.space.reduce(tensor)

    def inflate(self, tensor):
        return self.space.inflate(tensor)
