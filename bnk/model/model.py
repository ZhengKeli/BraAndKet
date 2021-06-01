import abc

import numpy as np

from .component import QComponent
from ..evolve import lindblad_evolve, schrodinger_evolve
from ..tensor import QTensor
from ..utils import structured_iter


class QModel(QComponent, abc.ABC):
    def __init__(self, children, hmt, gamma=None, deco=None, hb=1, initial=None, operators=None):
        self.hmt = hmt
        self.gamma = gamma
        self.deco = deco
        self.hb = hb

        operators = [*structured_iter(operators)]
        operators.extend(*structured_iter(hmt))
        operators.extend(*structured_iter(deco))

        super().__init__(children, initial, operators)

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
