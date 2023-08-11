import abc
import weakref
from typing import Iterable, Optional

from braandket.backend import Backend
from braandket.space import KetSpace
from braandket.tensor import PureStateTensor, StateTensor


# state

class QState:
    def __init__(self, tensor: StateTensor, related_systems: Iterable['QSystem'] = ()):
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

    @property
    def backend(self) -> Backend:
        return self.tensor.backend

    def _add_related_system(self, system: 'QSystem'):
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

class QSystem(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        pass

    @property
    @abc.abstractmethod
    def spaces(self) -> tuple[KetSpace, ...]:
        pass

    @property
    @abc.abstractmethod
    def state(self) -> QState:
        pass

    @abc.abstractmethod
    def _set_state(self, state: QState):
        pass

    @property
    def backend(self) -> Backend:
        return self.state.backend

    # compose

    def __matmul__(self, other: 'QSystem'):
        components = []

        if isinstance(self, QComposed) and self.name is None:
            components.extend(self)
        else:
            components.append(self)

        if isinstance(other, QComposed) and other.name is None:
            components.extend(other)
        else:
            components.append(other)

        return QComposed(*components)

    # str & repr

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name}>"


class QParticle(QSystem):
    def __init__(self, space: KetSpace, state_tensor: StateTensor):
        if space not in state_tensor.spaces:
            raise ValueError(f"Space {space} not included in the given state tensor!")
        self._space = space
        self._state = QState(state_tensor, (self,))

    @property
    def name(self) -> Optional[str]:
        return self._space.name

    @property
    def spaces(self) -> Iterable[KetSpace]:
        return (self._space,)

    @property
    def state(self) -> QState:
        return self._state

    def _set_state(self, state: QState):
        self._state = state


class QComposed(QSystem, Iterable[QSystem]):
    def __init__(self, *components: QSystem, name: Optional[str] = None):
        self._name = name
        self._components = components
        self._state = QState.prod(*(system.state for system in components))

    # system

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def spaces(self) -> tuple[KetSpace, ...]:
        return tuple(space for comp in self for space in comp.spaces)

    @property
    def state(self) -> QState:
        return self._state

    def _set_state(self, state: QState):
        self._state = state

    # components

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, item):
        return self._components[item]
