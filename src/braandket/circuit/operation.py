import abc
from typing import Optional, Union

from braandket.space import KetSpace
from braandket.tensor import NumericTensor, OperatorTensor, choose, take
from .system import QSystem


# measurement result

class MeasurementResult(abc.ABC):
    def __init__(self, operation: 'MeasurementOperation', *, name: Optional[str] = None):
        self._operation = operation
        self._name = name

    @property
    def operation(self) -> 'QOperation':
        return self._operation

    @property
    def name(self) -> Optional[str]:
        return self._name


class SingleMeasurementResult(MeasurementResult):
    def __init__(self,
            operation: 'MeasurementOperation',
            result: Union[int, NumericTensor],
            probability: Union[float, NumericTensor], *,
            name: Optional[str] = None
    ):
        super().__init__(operation, name=name)
        self._result = result
        self._probability = probability

    @property
    def operation(self) -> 'MeasurementOperation':
        return self._operation

    @property
    def result(self) -> NumericTensor:
        return self._result

    @property
    def probability(self) -> NumericTensor:
        return self._probability

    def __repr__(self) -> str:
        items = [f"result={self.result}", f"probability={self.probability}"]
        if self.name is not None:
            items.append(f"name={self.name}")
        return f"<{type(self).__name__} " + ", ".join(items) + ">"


class ComposedMeasurementResult(MeasurementResult):
    def __init__(self,
            operation: 'MeasurementOperation',
            *children: MeasurementResult,
            name: Optional[str] = None
    ):
        super().__init__(operation, name=name)
        self._children = children

    @property
    def children(self) -> tuple[MeasurementResult, ...]:
        return self._children


# operation

class QOperation(abc.ABC):
    def __call__(self, *systems: QSystem, name: Optional[str] = None) -> Optional[MeasurementResult]:
        return self.apply(*systems, name=name)

    @abc.abstractmethod
    def apply(self, *systems: QSystem, name: Optional[str] = None) -> Optional[MeasurementResult]:
        pass


class UnitaryOperation(QOperation, abc.ABC):
    @abc.abstractmethod
    def operator(self, *spaces: Union[KetSpace, tuple]) -> OperatorTensor:
        pass

    def apply(self, *systems: QSystem, name: Optional[str] = None) -> Optional[MeasurementResult]:
        spaces = tuple(system.spaces for system in systems)
        state = QSystem.prod(*systems).state

        with state.tensor.backend:
            operator = self.operator(*spaces)

        state.tensor = operator @ state.tensor
        return None


class MeasurementOperation(QOperation, abc.ABC):
    @abc.abstractmethod
    def operators(self, *spaces: Union[KetSpace, tuple]) -> tuple[OperatorTensor, ...]:
        pass

    def apply(self, *systems: QSystem, name: Optional[str] = None) -> SingleMeasurementResult:
        spaces = tuple(system.spaces for system in systems)
        state = QSystem.prod(*systems).state

        with state.tensor.backend:
            measure_operators = self.operators(*spaces)

        probs = []
        for measure_operator in measure_operators:
            measured_state = measure_operator @ state.tensor
            prob = measured_state.norm()
            probs.append(prob)

        measure_result = choose(probs)
        measured_state = take(measure_operators, measure_result) @ state.tensor
        measure_prob = take(probs, measure_result).as_numeric_tensor()

        state.tensor = measured_state.normalize()
        return SingleMeasurementResult(self, measure_result, measure_prob, name=name)

    def __call__(self, *systems: QSystem, name: Optional[str] = None) -> SingleMeasurementResult:
        return self.apply(*systems, name=name)


class DesiredMeasurementOperation(MeasurementOperation):
    def __init__(self, measurement: MeasurementOperation, decision: int):
        self._measurement = measurement
        self._decision = decision

    def operators(self, *spaces: Union[KetSpace, tuple]) -> tuple[OperatorTensor, ...]:
        return self._measurement.operators(*spaces)

    def operator(self, *spaces: Union[KetSpace, tuple]) -> OperatorTensor:
        return self.operators(*spaces)[self._decision]

    def apply(self, *systems: QSystem, name: Optional[str] = None) -> SingleMeasurementResult:
        spaces = tuple(system.spaces for system in systems)
        state = QSystem.prod(*systems).state

        with state.tensor.backend:
            measure_operator = self.operator(*spaces)

        measured_state = measure_operator @ state.tensor
        measure_prob = measured_state.norm()

        state.tensor = measured_state.normalize()
        return SingleMeasurementResult(self, self._decision, measure_prob, name=name)
