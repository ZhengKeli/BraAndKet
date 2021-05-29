import numpy as np

from ..evolve import lindblad_evolve, schrodinger_evolve
from ..tensor import QTensor


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
