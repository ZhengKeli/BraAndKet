import abc
from typing import Any, Generic, Iterable, Optional, Union

from braandket.backend import Backend, ValuesType, get_default_backend
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
        if isinstance(values, NumericTensor):
            return values
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        if spaces is None:
            spaces = ()
        return cls(values, spaces, backend or get_default_backend())

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        spaces = tuple(spaces)
        for space in spaces:
            if not isinstance(space, NumSpace):
                raise TypeError(f"NumTensor accepts only NumSpace, got {space}!")
        super().__init__(values, spaces, backend)

    @property
    def spaces(self) -> tuple[NumSpace, ...]:
        # noinspection PyTypeChecker
        return super().spaces

    # linear operations

    def _mul_expanded_num_spaces(self, other: 'QTensor') -> 'NumericTensor':
        return self @ other


class StateTensor(QTensor[ValuesType], Generic[ValuesType], abc.ABC):
    @abc.abstractmethod
    def trace(self, *spaces: Union[NumSpace, KetSpace]) -> 'StateTensor':
        pass

    @abc.abstractmethod
    def norm(self) -> 'NumericTensor':
        pass

    @abc.abstractmethod
    def normalize(self) -> 'StateTensor':
        pass

    @abc.abstractmethod
    def probabilities(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        pass


class PureStateTensor(StateTensor[ValuesType]):

    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'PureStateTensor':
        if isinstance(values, PureStateTensor):
            return values
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
    def spaces(self) -> tuple[Union[NumSpace, KetSpace], ...]:
        # noinspection PyTypeChecker
        return super().spaces

    @property
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        return tuple(space for space in self.spaces if isinstance(space, KetSpace))

    # special operations

    def trace(self, *spaces: Union[NumSpace, KetSpace]) -> Union['PureStateTensor', 'MixedStateTensor']:
        pass  # TODO

    def norm(self) -> 'NumericTensor':
        from .operations import abs
        return abs(NumericTensor.of(self.ct @ self))

    def normalize(self) -> 'PureStateTensor':
        from .operations import sqrt
        return self / sqrt(self.norm())

    def amplitudes(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        # return self.values(*spaces)
        # TODO
        pass

    def probabilities(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        # amplitudes = self.amplitudes(*spaces)
        # return self.backend.mul(self.backend.conj(amplitudes), amplitudes)
        # TODO
        pass


class MixedStateTensor(StateTensor[ValuesType]):
    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'MixedStateTensor':
        if isinstance(values, MixedStateTensor):
            return values
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
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        return self._ket_spaces

    def trace(self, *spaces: Union[NumSpace, KetSpace]) -> 'MixedStateTensor':
        values, spaces = self.values_and_spaces(*spaces)
        ket_axes, bra_axes = _index_spaces_pairs(spaces)
        return self.backend.trace(values, (ket_axes, bra_axes))

    def norm(self) -> 'NumericTensor':
        return NumericTensor.of(self.trace(*self.ket_spaces))

    def normalize(self) -> 'MixedStateTensor':
        return self / self.norm()

    def probabilities(self, *spaces: Union[NumSpace, KetSpace]) -> ValuesType:
        pass  # TODO


class OperatorTensor(QTensor[ValuesType]):
    @classmethod
    def of(cls,
            values: Union[QTensor, Any],
            spaces: Optional[Iterable[Space]] = None, *,
            backend: Optional[Backend] = None
    ) -> 'OperatorTensor':
        if isinstance(values, OperatorTensor):
            return values
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces, values._backend)
        return cls(values, spaces, backend or get_default_backend())

    @classmethod
    def from_matrix(cls,
            matrix: ValuesType,
            ket_spaces: Iterable[KetSpace],
            num_spaces: Iterable[KetSpace] = (), *,
            backend: Optional[Backend] = None
    ) -> 'OperatorTensor':
        num_spaces = tuple(num_spaces)
        ket_spaces = tuple(ket_spaces)
        bra_spaces = tuple(space.ct for space in ket_spaces)
        spaces = num_spaces + ket_spaces + bra_spaces
        backend = backend or get_default_backend()
        shape = tuple(space.n for space in spaces)
        values = backend.reshape(matrix, shape)
        return cls(values, spaces, backend)

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        spaces = tuple(spaces)
        ket_spaces = _match_spaces_pairs(spaces)
        super().__init__(values, spaces, backend)
        self._ket_spaces = ket_spaces

    @property
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        return self._ket_spaces

    # linear operations

    def __add__(self, other) -> 'OperatorTensor':
        return OperatorTensor.of(super().__add__(other))

    def __mul__(self, other) -> 'OperatorTensor':
        return OperatorTensor.of(super().__mul__(other))

    @classmethod
    def _expand_tensors_for_addsub(cls, tensor0: 'QTensor', tensor1: 'QTensor') -> tuple['QTensor', 'QTensor']:
        assert isinstance(tensor0, OperatorTensor)
        assert isinstance(tensor1, OperatorTensor)
        tensor0_ket_spaces = tensor0.ket_spaces
        tensor1_ket_spaces = tensor1.ket_spaces

        from .operations import _expand_with_identities
        tensor0 = _expand_with_identities(tensor0, *tensor1_ket_spaces)
        tensor1 = _expand_with_identities(tensor1, *tensor0_ket_spaces)
        return super()._expand_tensors_for_addsub(tensor0, tensor1)

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


def _match_spaces_pairs(spaces: Iterable[Space]) -> tuple[KetSpace, ...]:
    spaces = tuple(spaces)
    ket_axes, bra_axes = _index_spaces_pairs(spaces)
    return tuple(spaces[axis] for axis in ket_axes)
