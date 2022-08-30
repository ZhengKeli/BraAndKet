from typing import Optional, Union

import numpy as np

from braandket.backends import Backend
from braandket.space import KetSpace
from braandket.tensor import NumericTensor, OperatorTensor, QTensor, cos, exp, sin
from .operation import MeasurementOperation, UnitaryOperation


class _M(MeasurementOperation):
    # noinspection PyMethodOverriding
    def _measure_operators(self, space: KetSpace, *, backend: Backend) -> tuple[OperatorTensor, ...]:
        # noinspection PyTypeChecker
        return space.projector(0, backend=backend), space.projector(1, backend=backend)


class _X(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        return qubit.operator(0, 1, backend=backend) + qubit.operator(1, 0, backend=backend)


class _Y(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        return (- qubit.operator(0, 1, backend=backend)
                + qubit.operator(1, 0, backend=backend)) * 1j


class _Z(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        return qubit.projector(0, backend=backend) - qubit.projector(1, backend=backend)


class _H(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        return (qubit.operator(0, 0, backend=backend) + qubit.operator(0, 1, backend=backend) +
                qubit.operator(1, 0, backend=backend) + qubit.operator(1, 1, backend=backend)) / backend.sqrt(2)


class _CX(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, control: KetSpace, target: KetSpace, *, backend: Backend) -> OperatorTensor:
        return control.projector(0, backend=backend) @ target.identity(backend=backend) + \
               control.projector(1, backend=backend) @ X._unitary_operator(target, backend=backend)


class _CY(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, control: KetSpace, target: KetSpace, *, backend: Backend) -> OperatorTensor:
        return control.projector(0, backend=backend) @ target.identity(backend=backend) + \
               control.projector(1, backend=backend) @ Y._unitary_operator(target, backend=backend)


class _CZ(UnitaryOperation):
    # noinspection PyMethodOverriding
    def _unitary_operator(self, control: KetSpace, target: KetSpace, *, backend: Backend) -> OperatorTensor:
        return control.projector(0, backend=backend) @ target.identity(backend=backend) + \
               control.projector(1, backend=backend) @ target.identity(backend=backend) * (-1)


M = _M()
X = _X()
Y = _Y()
Z = _Z()
H = _H()
CX = _CX()
CY = _CY()
CZ = _CZ()
CNOT = CX


class Rx(UnitaryOperation):
    def __init__(self, theta: Union[QTensor, np.ndarray, float], *, backend: Optional[Backend] = None):
        self._theta: NumericTensor = NumericTensor.of(theta, (), backend=backend)

    @property
    def theta(self) -> NumericTensor:
        return self._theta

    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        half_theta = self.theta / 2
        return qubit.operator(0, 0, backend=backend) * (cos(half_theta) * +1) + \
               qubit.operator(0, 1, backend=backend) * (sin(half_theta) * -1j) + \
               qubit.operator(1, 0, backend=backend) * (sin(half_theta) * -1j) + \
               qubit.operator(1, 1, backend=backend) * (cos(half_theta) * +1)


class Ry(UnitaryOperation):

    def __init__(self, theta: float):
        self._theta = theta

    @property
    def theta(self) -> float:
        return self._theta

    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        half_theta = self.theta / 2
        return qubit.operator(0, 0, backend=backend) * (cos(half_theta) * +1) + \
               qubit.operator(0, 1, backend=backend) * (sin(half_theta) * -1) + \
               qubit.operator(1, 0, backend=backend) * (sin(half_theta) * -1) + \
               qubit.operator(1, 1, backend=backend) * (cos(half_theta) * +1)


class Rz(UnitaryOperation):
    def __init__(self, theta: float):
        self._theta = theta

    @property
    def theta(self) -> float:
        return self._theta

    # noinspection PyMethodOverriding
    def _unitary_operator(self, qubit: KetSpace, *, backend: Backend) -> OperatorTensor:
        return qubit.projector(0, backend=backend) * exp(self.theta * -0.5j) + \
               qubit.projector(1, backend=backend) * exp(self.theta * +0.5j)
