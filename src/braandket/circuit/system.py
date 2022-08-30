import abc
from typing import Iterable, Optional
from weakref import WeakSet

from braandket.space import KetSpace
from braandket.tensor import PureStateTensor


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


class QSystem(abc.ABC):
    def __init__(self, state: QState, spaces: Iterable[KetSpace]):
        self._state = state
        self._spaces = frozenset(spaces)

        # noinspection PyProtectedMember
        self._state._systems.add(self)

    @property
    def state(self) -> 'QState':
        return self._state

    @property
    def spaces(self) -> frozenset[KetSpace]:
        return self._spaces

    def __matmul__(self, other: 'QSystem'):
        return QCompose(self, other)

    @classmethod
    def prod(cls, system0: 'QSystem', *systems: 'QSystem') -> 'QSystem':
        return cls.prod(system0 @ systems[0], *systems[1:]) \
            if len(systems) > 0 else system0


class QCompose(QSystem):
    def __init__(self, *systems: QSystem):
        super().__init__(
            state=QState.prod(*(system.state for system in systems)),
            spaces=(space for system in systems for space in system.spaces))


class QParticle(QSystem, KetSpace):
    def __init__(self, n: int, name: Optional[str] = None):
        KetSpace.__init__(self, n, name)
        state = QState(self.eigenstate(0), ())
        QSystem.__init__(self, state, (self,))


class Qubit(QParticle):
    def __init__(self, name: Optional[str] = None):
        super().__init__(2, name)
