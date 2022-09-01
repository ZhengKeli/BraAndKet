from typing import Optional, Union

import numpy as np

from braandket.space import KetSpace
from braandket.tensor import NumericTensor, OperatorTensor, QTensor, cos, exp, sin, sqrt
from .operation import DesiredMeasurementOperation, MeasurementOperation, UnitaryOperation
from .system import QParticle


# systems

class Qubit(QParticle):
    def __init__(self, name: Optional[str] = None):
        super().__init__(2, name)


# measurements

class _M(MeasurementOperation):
    # noinspection PyMethodOverriding
    def operators(self, space: KetSpace) -> tuple[OperatorTensor, ...]:
        # noinspection PyTypeChecker
        return space.projector(0), space.projector(1)


class D(DesiredMeasurementOperation):
    def __init__(self, decision: int):
        super().__init__(M, decision)


# unitary operations

class _X(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        return qubit.operator(0, 1) + qubit.operator(1, 0)


class _Y(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        return (- qubit.operator(0, 1)
                + qubit.operator(1, 0)) * 1j


class _Z(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        return qubit.projector(0) - qubit.projector(1)


class _H(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        return (qubit.operator(0, 0) + qubit.operator(0, 1) +
                qubit.operator(1, 0) + qubit.operator(1, 1)) / sqrt(2)


class _CX(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, control: KetSpace, target: KetSpace) -> OperatorTensor:
        return control.projector(0) @ target.identity() + \
               control.projector(1) @ X.operator(target)


class _CY(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, control: KetSpace, target: KetSpace) -> OperatorTensor:
        return control.projector(0) @ target.identity() + \
               control.projector(1) @ Y.operator(target)


class _CZ(UnitaryOperation):
    # noinspection PyMethodOverriding
    def operator(self, control: KetSpace, target: KetSpace) -> OperatorTensor:
        return control.projector(0) @ target.identity() + \
               control.projector(1) @ target.identity() * (-1)


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
    def __init__(self, theta: Union[QTensor, np.ndarray, float]):
        self._theta: NumericTensor = NumericTensor.of(theta, ())

    @property
    def theta(self) -> NumericTensor:
        return self._theta

    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        half_theta = self.theta / 2
        return qubit.operator(0, 0) * (cos(half_theta) * +1) + \
               qubit.operator(0, 1) * (sin(half_theta) * -1j) + \
               qubit.operator(1, 0) * (sin(half_theta) * -1j) + \
               qubit.operator(1, 1) * (cos(half_theta) * +1)


class Ry(UnitaryOperation):
    def __init__(self, theta: Union[QTensor, np.ndarray, float]):
        self._theta: NumericTensor = NumericTensor.of(theta, ())

    @property
    def theta(self) -> NumericTensor:
        return self._theta

    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        half_theta = self.theta / 2
        return qubit.operator(0, 0) * (cos(half_theta) * +1) + \
               qubit.operator(0, 1) * (sin(half_theta) * -1) + \
               qubit.operator(1, 0) * (sin(half_theta) * -1) + \
               qubit.operator(1, 1) * (cos(half_theta) * +1)


class Rz(UnitaryOperation):
    def __init__(self, theta: Union[QTensor, np.ndarray, float]):
        self._theta: NumericTensor = NumericTensor.of(theta, ())

    @property
    def theta(self) -> NumericTensor:
        return self._theta

    # noinspection PyMethodOverriding
    def operator(self, qubit: KetSpace) -> OperatorTensor:
        return qubit.projector(0) * exp(self.theta * -0.5j) + \
               qubit.projector(1) * exp(self.theta * +0.5j)
