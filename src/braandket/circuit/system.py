import abc
from typing import Iterable, Optional, Union
from weakref import WeakSet

from braandket.space import KetSpace
from braandket.tensor import PureStateTensor


# state

class QState:
    def __init__(self, tensor: PureStateTensor, systems: Iterable['QSystem']):
        self._tensor = tensor
        self._systems = WeakSet(systems)

    @property
    def tensor(self) -> PureStateTensor:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: PureStateTensor):
        self._tensor = tensor

    def __matmul__(self, other: 'QState') -> 'QState':
        if other is self:
            return self
        new_tensor = self._tensor @ other._tensor
        new_systems = (*self._systems, *other._systems)
        new_state = QState(new_tensor, new_systems)
        for system in new_systems:
            system._state = new_state
        return new_state

    @classmethod
    def prod(cls, state0: 'QState', *states: 'QState') -> 'QState':
        return cls.prod(state0 @ states[0], *states[1:]) \
            if len(states) > 0 else state0


# system

class QSystem(abc.ABC):
    def __init__(self, state: QState):
        self._state = state

        # noinspection PyProtectedMember
        self._state._systems.add(self)

    @property
    def state(self) -> 'QState':
        return self._state

    @property
    @abc.abstractmethod
    def spaces(self) -> Union[KetSpace, tuple]:
        pass

    def __matmul__(self, other: 'QSystem'):
        return QCompose(self, other)

    @classmethod
    def prod(cls, system0: 'QSystem', *systems: 'QSystem') -> 'QSystem':
        return cls.prod(system0 @ systems[0], *systems[1:]) \
            if len(systems) > 0 else system0


class QCompose(QSystem):
    def __init__(self, *children: QSystem):
        super().__init__(QState.prod(*(system.state for system in children)))
        self._children = tuple(children)

    @property
    def children(self) -> tuple[QSystem]:
        return self._children

    @property
    def spaces(self) -> Union[KetSpace, tuple]:
        return tuple(child.spaces for child in self._children)


class QParticle(QSystem, KetSpace):
    def __init__(self, n: int, name: Optional[str] = None):
        KetSpace.__init__(self, n, name)
        QSystem.__init__(self, QState(self.eigenstate(0), ()))

    @property
    def spaces(self) -> Union[KetSpace, tuple]:
        return self
