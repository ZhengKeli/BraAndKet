import abc
import weakref
from typing import Iterable, Optional

from braandket.backend import Backend
from braandket.space import KetSpace
from braandket.tensor import PureStateTensor, StateTensor


# state

class QState:
    def __init__(self, tensor: StateTensor, related_systems: Iterable['QModel'] = ()):
        self._tensor = tensor
        self._related_systems = weakref.WeakSet(related_systems)
        for system in self._related_systems:
            # noinspection PyProtectedMember
            system._set_state(self)

    @property
    def tensor(self) -> StateTensor:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: StateTensor):
        self._tensor = tensor

    def _add_related_system(self, system: 'QModel'):
        self._related_systems.add(system)

    def __matmul__(self, other: 'QState') -> 'QState':
        if other is self:
            return self
        new_tensor = self._tensor @ other._tensor
        new_related_systems = (*self._related_systems, *other._related_systems)
        return QState(new_tensor, new_related_systems)

    @classmethod
    def prod(cls, *states: 'QState') -> 'QState':
        if len(states) == 0:
            return QState(PureStateTensor.of((), ()))
        if len(states) == 1:
            return states[0]
        return cls.prod(states[0] @ states[1], *states[2:])


# system

class QModel(abc.ABC):
    def __init__(self, *, name: Optional[str] = None):
        self._name = name
        self._state = None

    @property
    def name(self) -> Optional[str]:
        return self._name

    # particles

    @property
    @abc.abstractmethod
    def particles(self) -> frozenset['QParticle']:
        pass

    # state

    @property
    def state(self) -> QState:
        if self._state is None:
            self._state = self._init_state()
            # noinspection PyProtectedMember
            self._state._add_related_system(self)
        return self._state

    @abc.abstractmethod
    def _init_state(self) -> QState:
        pass

    def _set_state(self, state: QState):
        self._state = state

    # compose

    def __matmul__(self, other: 'QModel'):
        children = []

        if isinstance(self, QComposed) and self.name is None:
            children.extend(self)
        else:
            children.append(self)

        if isinstance(other, QComposed) and other.name is None:
            children.extend(other)
        else:
            children.append(other)

        return QComposed(*children)


class QParticle(QModel, KetSpace):
    def __init__(self, n: int, *, name: Optional[str] = None, backend: Optional[Backend] = None):
        QModel.__init__(self, name=name)
        KetSpace.__init__(self, n, name)
        self._backend = backend

    # particles

    @property
    def particles(self) -> frozenset['QParticle']:
        return frozenset((self,))

    # state

    def _init_state(self) -> QState:
        return QState(self.eigenstate(0, backend=self._backend))


class QComposed(QModel, Iterable[QModel]):
    def __init__(self, *children: QModel, name: Optional[str] = None):
        super().__init__(name=name)
        self._children = tuple(children)

    # children

    @property
    def children(self) -> tuple[QModel]:
        return self._children

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def __getitem__(self, item):
        return self._children[item]

    # particles

    @property
    def particles(self) -> frozenset['QParticle']:
        return frozenset.union(*(child.particles for child in self._children))

    # state

    def _init_state(self) -> QState:
        return QState.prod(*(system.state for system in self._children))
