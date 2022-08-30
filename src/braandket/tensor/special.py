from typing import Any, Iterable, Optional, Union

from braandket.backends import Backend, ValuesType, get_default_backend
from braandket.space import KetSpace, NumSpace, Space
from .tensor import QTensor


# special tensors

class NumericTensor(QTensor[ValuesType]):

    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Space]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'NumericTensor':
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        return cls(values, spaces, backend or get_default_backend())

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        spaces = tuple(spaces)
        for space in spaces:
            if not isinstance(space, NumSpace):
                raise TypeError(f"NumTensor accepts only NumSpace, got {space}!")
        super().__init__(values, spaces, backend)

    @property
    def spaces(self) -> frozenset[NumSpace, ...]:
        # noinspection PyTypeChecker
        return super().spaces

    # linear operations

    def _mul_expanded_num_spaces(self, other: 'QTensor') -> 'NumericTensor':
        return self @ other


class PureStateTensor(QTensor[ValuesType]):

    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'PureStateTensor':
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        return cls(values, spaces, backend or get_default_backend())

    def __init__(self, values: Any, spaces: Iterable[Union[NumSpace, KetSpace]], backend: Backend):
        spaces = tuple(spaces)
        for space in spaces:
            if not isinstance(space, (NumSpace, KetSpace)):
                raise TypeError(f"PureStateTensor accepts only KetSpace and NumSpace, got {space}!")
        super().__init__(values, spaces, backend)

    @property
    def spaces(self) -> frozenset[KetSpace, ...]:
        # noinspection PyTypeChecker
        return super().spaces

    # special operations

    def normalize(self) -> 'PureStateTensor':
        num_spaces = tuple(space for space in self.spaces if isinstance(space, NumSpace))
        ket_spaces = tuple(space for space in self.spaces if isinstance(space, KetSpace))
        values = self.values(*num_spaces, *ket_spaces)

        prob = self.ct @ self
        prob_values = self.backend.expand(
            prob.values(*num_spaces),
            axes=range(len(num_spaces), len(num_spaces) + len(ket_spaces)))

        new_values = self.backend.div(values, self.backend.sqrt(prob_values))
        new_spaces = [*num_spaces, *ket_spaces]
        return self.of(new_values, new_spaces)

    def amplitudes(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        # return self.values(*spaces)
        # TODO
        pass

    def probabilities(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        # amplitudes = self.amplitudes(*spaces)
        # return self.backend.mul(self.backend.conj(amplitudes), amplitudes)
        # TODO
        pass


class MixedStateTensor(QTensor[ValuesType]):
    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'MixedStateTensor':
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        return cls(values, spaces, backend or get_default_backend())

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        spaces = tuple(spaces)
        ket_spaces = _match_spaces_pairs(spaces)
        super().__init__(values, spaces, backend)
        self._ket_spaces = ket_spaces

    @property
    def ket_spaces(self) -> frozenset[KetSpace]:
        return self._ket_spaces

    def trace(self, *spaces: Union[NumSpace, KetSpace]) -> 'MixedStateTensor':
        values, spaces = self.values_and_spaces(*spaces)
        ket_axes, bra_axes = _index_spaces_pairs(spaces)
        return self.backend.trace(values, (ket_axes, bra_axes))

    def normalize(self) -> 'MixedStateTensor':
        return self / self.trace(*self.ket_spaces)

    def probabilities(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        # TODO
        pass


class OperatorTensor(QTensor[ValuesType]):
    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'OperatorTensor':
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        return cls(values, spaces, backend or get_default_backend())

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        spaces = tuple(spaces)
        ket_spaces = _match_spaces_pairs(spaces)
        super().__init__(values, spaces, backend)
        self._ket_spaces = ket_spaces

    @property
    def ket_spaces(self) -> frozenset[KetSpace]:
        return self._ket_spaces

    # linear operations

    def __add__(self, other) -> 'OperatorTensor':
        return OperatorTensor.of(super().__add__(other))

    def __mul__(self, other) -> 'OperatorTensor':
        return OperatorTensor.of(super().__mul__(other))

    @classmethod
    def _expand_tensors_for_add(cls, tensor0: 'QTensor', tensor1: 'QTensor') -> tuple['QTensor', 'QTensor']:
        assert isinstance(tensor0, OperatorTensor)
        assert isinstance(tensor1, OperatorTensor)
        tensor0_ket_spaces = tensor0.ket_spaces
        tensor1_ket_spaces = tensor1.ket_spaces

        from .operations import _expand_with_identities
        tensor0 = _expand_with_identities(tensor0, *tensor1_ket_spaces)
        tensor1 = _expand_with_identities(tensor1, *tensor0_ket_spaces)
        return super()._expand_tensors_for_add(tensor0, tensor1)

    # spaces operations

    @property
    def ct(self) -> 'OperatorTensor':
        return OperatorTensor.of(super().ct)

    def __matmul__(self, other) -> 'QTensor':
        tensor = super().__matmul__(other)
        if isinstance(other, OperatorTensor):
            tensor = OperatorTensor.of(tensor)
        return tensor


# utils

def _index_spaces_pairs(spaces: Iterable[Space]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    spaces = tuple(spaces)
    ket_axes = []
    bra_axes = []
    for axis, space in enumerate(spaces):
        if isinstance(space, KetSpace):
            ket_axes.append(axis)
            bra_axes.append(spaces.index(space.ct))
    return tuple(ket_axes), tuple(bra_axes)


def _match_spaces_pairs(spaces: Iterable[Space]) -> frozenset[KetSpace, ...]:
    spaces = tuple(spaces)
    ket_axes, bra_axes = _index_spaces_pairs(spaces)
    return frozenset(spaces[axis] for axis in ket_axes)
