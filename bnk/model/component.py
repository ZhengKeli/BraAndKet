import abc

import numpy as np

from ..space import KetSpace
from ..utils import prod


class QComponent(abc.ABC):
    def __init__(self, children, initial=None, operators=None):
        if not isinstance(children, dict):
            children = dict(enumerate(tuple(children)))

        for child in children.values():
            if not isinstance(child, (KetSpace, QComponent)):
                raise TypeError(f"parameter children must be instances of KetSpace or QComponent, got {child}")

        self._children = children
        self._initial = None if initial is None else tuple(initial)
        self._operators = None if operators is None else tuple(operators)

    @property
    def children(self):
        return self._children

    @property
    def initial(self):
        return self._initial

    @property
    def operators(self):
        return self._operators

    @property
    def spaces(self):
        for child in self.children.values():
            if isinstance(child, KetSpace):
                yield child
            elif isinstance(child, QComponent):
                for spaces in child.spaces:
                    yield spaces

    def eigenstate(self, indices, *args, sparse=False, dtype=np.float32, **kwargs):
        if not isinstance(indices, dict):
            indices = dict(enumerate(tuple(indices)))

        eigenstates = []
        for k, index in indices.items():
            child = self.children[k]
            if isinstance(child, KetSpace):
                eigenstate = child.eigenstate(index)
            elif isinstance(child, QComponent):
                eigenstate = child.eigenstate(index, sparse=sparse, dtype=dtype)
            else:
                assert False
            eigenstates.append(eigenstate)
        return prod(eigenstates)
