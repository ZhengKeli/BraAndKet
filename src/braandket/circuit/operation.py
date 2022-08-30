import abc
from typing import Iterable

from braandket.backends import Backend
from braandket.space import KetSpace
from braandket.tensor import OperatorTensor, abs, choose, take
from .system import QState, QSystem


class QOperation(abc.ABC):
    def __call__(self, *systems: QSystem) -> tuple[QSystem]:
        return self.apply(*systems)

    def apply(self, *systems: QSystem) -> tuple[QSystem]:
        self.apply_with_measurement(*systems)
        return systems

    def apply_with_measurement(self, *systems: QSystem):
        system = systems[0] if len(systems) == 1 else QSystem.prod(*systems)
        results = self._apply(system.state, *system.spaces)
        return results

    @abc.abstractmethod
    def _apply(self, state: QState, *spaces: Iterable[KetSpace]):
        pass


class UnitaryOperation(QOperation, abc.ABC):
    @abc.abstractmethod
    def _unitary_operator(self, *spaces: KetSpace, backend: Backend) -> OperatorTensor:
        pass

    def _apply(self, state: QState, *spaces: Iterable[KetSpace]):
        op_tensor = self._unitary_operator(*spaces, backend=state.tensor.backend)
        state.tensor = op_tensor @ state.tensor


class MeasurementOperation(QOperation, abc.ABC):
    @abc.abstractmethod
    def _measure_operators(self, *spaces: KetSpace, backend: Backend) -> tuple[OperatorTensor, ...]:
        pass

    def _apply(self, state: QState, *spaces: Iterable[KetSpace]):
        backend = state.tensor.backend
        measure_operators = self._measure_operators(*spaces, backend=backend)

        probs = []
        for measure_operator in measure_operators:
            measured_state = measure_operator @ state.tensor
            prob = abs((measured_state.ct @ measured_state)).as_numeric_tensor()
            probs.append(prob)

        measure_result = choose(probs)
        measure_prob = take(probs, measure_result)
        return measure_result, measure_prob
