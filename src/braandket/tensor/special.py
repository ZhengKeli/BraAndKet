import abc
from typing import Any, Generic, Iterable, Optional, Union, overload

from braandket.backend import BackendValue, get_default_backend
from braandket.space import BraSpace, HSpace, KetSpace, NumSpace, Space
from .tensor import QTensor


# special tensors

class NumericTensor(QTensor[BackendValue]):

    @classmethod
    def of(cls,
        values: Union[QTensor, Any],
        spaces: Optional[Iterable[Space]] = None,
    ) -> 'NumericTensor':
        if isinstance(values, NumericTensor):
            return values
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces)
        if spaces is None:
            spaces = ()
        return cls(values, spaces)

    def __init__(self, values: Any, spaces: Iterable[Space]):
        spaces = tuple(spaces)
        for space in spaces:
            if not isinstance(space, NumSpace):
                raise TypeError(f"NumTensor accepts only NumSpace, got {space}!")
        super().__init__(values, spaces)

    @property
    def spaces(self) -> tuple[NumSpace, ...]:
        # noinspection PyTypeChecker
        return super().spaces

    # linear operations

    def _mul_expanded_num_spaces(self, other: 'QTensor') -> 'NumericTensor':
        return self @ other


class StateTensor(QTensor[BackendValue], Generic[BackendValue], abc.ABC):
    @property
    @abc.abstractmethod
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        pass

    # norm

    @abc.abstractmethod
    def norm(self) -> 'NumericTensor':
        pass

    @abc.abstractmethod
    def normalize(self) -> 'StateTensor':
        pass

    @abc.abstractmethod
    def norm_and_normalize(self) -> tuple['NumericTensor', 'StateTensor']:
        return self.norm(), self.normalize()

    # component

    @abc.abstractmethod
    def component(self, indices: Union[Iterable[tuple[KetSpace, int]], dict[KetSpace, int]]) -> 'StateTensor':
        pass

    @abc.abstractmethod
    def trace(self, *spaces: KetSpace) -> 'StateTensor':
        pass

    def remain(self, *spaces: KetSpace) -> 'StateTensor':
        all_ket_spaces = frozenset(self.ket_spaces)
        remain_ket_spaces_set = frozenset(spaces)
        traced_ket_spaces = all_ket_spaces - remain_ket_spaces_set
        return self.trace(*traced_ket_spaces)

    @abc.abstractmethod
    def probabilities(self,
        *spaces: Union[NumSpace, KetSpace, tuple[Union[NumSpace, KetSpace], Union[int, None]]]
    ) -> BackendValue:
        pass

    # measurement

    @overload
    @abc.abstractmethod
    def measure(self,
        space: Union[KetSpace, tuple[KetSpace, int]],
    ) -> tuple[BackendValue, BackendValue, 'StateTensor']:
        pass

    @overload
    @abc.abstractmethod
    def measure(self,
        space0: Union[KetSpace, tuple[KetSpace, int]],
        space1: Union[KetSpace, tuple[KetSpace, int]],
        *spaces: Union[KetSpace, tuple[KetSpace, int]]
    ) -> tuple[BackendValue, BackendValue, 'StateTensor']:
        pass

    @overload
    @abc.abstractmethod
    def measure(self,
        spaces: Iterable[Union[KetSpace, tuple[KetSpace, int]]]
    ) -> tuple[BackendValue, BackendValue, 'StateTensor']:
        pass

    @abc.abstractmethod
    def measure(self,
        *args: Union[KetSpace, tuple[KetSpace, int], Iterable[Union[KetSpace, tuple[KetSpace, int]]]]
    ) -> tuple[BackendValue, BackendValue, 'StateTensor']:
        pass

    @classmethod
    def _formalize_measure_args(cls,
        *args: Union[KetSpace, tuple[KetSpace, int], Iterable[Union[KetSpace, tuple[KetSpace, int]]]]
    ) -> tuple[bool, Iterable[KetSpace], Optional[Iterable[int]]]:
        if len(args) == 1:
            if isinstance(args[0], KetSpace):
                return True, args, None
            elif isinstance(args[0], tuple) and len(args[0]) == 2 and isinstance(args[0][0], KetSpace):
                spaces, results = zip(*args)
                return True, spaces, results
            elif isinstance(args[0], Iterable):
                args = tuple(args[0])
        if all(isinstance(arg, KetSpace) for arg in args):
            return False, args, None
        elif all(isinstance(arg, tuple) for arg in args):
            spaces, results = zip(*args)
            return False, spaces, results
        else:
            raise NotImplementedError("Mixing normal and desired measurement is currently not supported!")


class PureStateTensor(StateTensor[BackendValue]):

    @classmethod
    def of(cls,
        values: Union[QTensor, Any],
        spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None,
    ) -> 'PureStateTensor':
        if isinstance(values, PureStateTensor):
            return values
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces)
        return cls(values, spaces)

    def __init__(self, values: Any, spaces: Iterable[Union[NumSpace, KetSpace]]):
        spaces = tuple(spaces)
        for space in spaces:
            if not isinstance(space, (NumSpace, KetSpace)):
                raise TypeError(f"PureStateTensor accepts only KetSpace and NumSpace, got {space}!")
        super().__init__(values, spaces)
        self._ket_spaces = tuple(space for space in spaces if isinstance(space, KetSpace))

    @property
    def spaces(self) -> tuple[Union[NumSpace, KetSpace], ...]:
        # noinspection PyTypeChecker
        return super().spaces

    @property
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        return self._ket_spaces

    # norm

    def norm(self) -> 'NumericTensor':
        from .operations import abs
        return abs(NumericTensor.of(self.ct @ self))

    def normalize(self) -> 'PureStateTensor':
        return self.norm_and_normalize()[1]

    def norm_and_normalize(self) -> tuple['NumericTensor', 'PureStateTensor']:
        from .operations import sqrt
        norm = self.norm()
        normalized = self / sqrt(norm)
        return norm, normalized

    # component

    def component(self, indices: Union[Iterable[tuple[KetSpace, int]], dict[KetSpace, int]]) -> 'PureStateTensor':
        if isinstance(indices, dict):
            indices = tuple(indices.items())
        else:
            indices = tuple(indices)
        return PureStateTensor.of(self[indices])

    def trace(self, *spaces: KetSpace) -> 'MixedStateTensor':
        return MixedStateTensor.of(self @ self.ct).trace(*spaces)

    def amplitudes(self,
        *spaces: Union[NumSpace, KetSpace, tuple[Union[NumSpace, KetSpace], Union[int, None]]]
    ) -> BackendValue:
        return self.values(*spaces)

    def probabilities(self,
        *spaces: Union[NumSpace, KetSpace, tuple[Union[NumSpace, KetSpace], Union[int, None]]]
    ) -> BackendValue:
        amplitudes = self.amplitudes(*spaces)
        probs = self.backend.mul(self.backend.conj(amplitudes), amplitudes)
        probs = self.backend.abs(probs)
        return probs

    # measurement

    def measure(self,
        *args: Union[KetSpace, tuple[KetSpace, int], Iterable[Union[KetSpace, tuple[KetSpace, int]]]]
    ) -> tuple[Union[BackendValue, Iterable[BackendValue]], BackendValue, 'StateTensor']:
        single, spaces, results = self._formalize_measure_args(*args)

        measure_axes = tuple(self.spaces.index(space) for space in spaces)
        reduced_axes = tuple(axis for axis, space in enumerate(self.spaces)
                             if isinstance(space, KetSpace) and axis not in measure_axes)
        batches_axes = tuple(axis for axis, space in enumerate(self.spaces)
                             if axis not in measure_axes and axis not in reduced_axes)

        results, prob, state_value = self.backend.measure_pure_state(
            self.values(), batches_axes, reduced_axes, measure_axes, results)

        results = results[..., 0] if single else results
        state = PureStateTensor.of(state_value, [
            *(self.spaces[axis] for axis in batches_axes),
            *spaces,
            *(self.spaces[axis] for axis in reduced_axes),
        ], backend=self.backend)

        return results, prob, state


class MixedStateTensor(StateTensor[BackendValue]):
    @classmethod
    def of(cls,
        values: Union[QTensor, Any],
        spaces: Optional[Iterable[Union[NumSpace, KetSpace]]] = None,
    ) -> 'MixedStateTensor':
        if isinstance(values, MixedStateTensor):
            return values
        if isinstance(values, PureStateTensor):
            return cls.of(values @ values.ct)
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces)
        return cls(values, spaces)

    def __init__(self, values: Any, spaces: Iterable[Space]):
        spaces = tuple(spaces)
        ket_spaces = _match_spaces_pairs(spaces)
        super().__init__(values, spaces)
        self._ket_spaces = ket_spaces

    @property
    def ket_spaces(self) -> tuple[KetSpace, ...]:
        return self._ket_spaces

    # norm

    def norm(self) -> 'NumericTensor':
        return NumericTensor.of(self.trace(*self.ket_spaces))

    def normalize(self) -> 'MixedStateTensor':
        return self.norm_and_normalize()[1]

    def norm_and_normalize(self) -> tuple['NumericTensor', 'MixedStateTensor']:
        norm = self.norm()
        normalized = self / norm
        return norm, normalized

    # component

    def component(self, indices: Union[Iterable[tuple[KetSpace, int]], dict[KetSpace, int]]) -> 'MixedStateTensor':
        if isinstance(indices, dict):
            ket_indices = tuple(indices.items())
        else:
            ket_indices = tuple(indices)
        bra_indices = tuple((ket_space.ct, index) for ket_space, index in ket_indices)
        return MixedStateTensor.of(self[ket_indices + bra_indices])

    def trace(self, *spaces: KetSpace) -> 'MixedStateTensor':
        if any(not isinstance(space, KetSpace) for space in spaces):
            raise TypeError(f"trace only accepts KetSpace, got {spaces}!")

        ket_axes = tuple(self.spaces.index(space) for space in spaces)
        bra_axes = tuple(self.spaces.index(space.ct) for space in spaces)
        new_value = self.backend.trace(self._values, (ket_axes, bra_axes))

        new_spaces = []
        spaces_set = set(spaces)
        for space in self.spaces:
            if not isinstance(space, HSpace):
                new_spaces.append(space)
            if isinstance(space, KetSpace):
                if space not in spaces_set:
                    new_spaces.append(space)
            if isinstance(space, BraSpace):
                if space.ct not in spaces_set:
                    new_spaces.append(space)

        return MixedStateTensor(new_value, new_spaces, backend=self.backend)

    def probabilities(self,
        *spaces: Union[NumSpace, KetSpace, tuple[Union[NumSpace, KetSpace], Union[int, None]]]
    ) -> BackendValue:
        bra_spaces = []
        for space_or_pair in spaces:
            if isinstance(space_or_pair, Space):
                space, index = space_or_pair, None
            else:
                space, index = space_or_pair
            if isinstance(space, KetSpace):
                if index is None:
                    bra_spaces.append(space.ct)
                else:
                    bra_spaces.append((space.ct, index))
        return self.values(*spaces, *bra_spaces)

    # measurement

    def measure(self,
        *args: Union[KetSpace, tuple[KetSpace, int], Iterable[Union[KetSpace, tuple[KetSpace, int]]]]
    ) -> tuple[Union[BackendValue, Iterable[BackendValue]], BackendValue, 'StateTensor']:
        single, spaces, results = self._formalize_measure_args(*args)

        measure_axes = tuple((self.spaces.index(space), self.spaces.index(space.ct)) for space in spaces)
        reduced_axes = tuple((self.spaces.index(space), self.spaces.index(space.ct)) for space in self.spaces
                             if isinstance(space, KetSpace) and space not in spaces)

        measure_axes_set = set(axis for axes in measure_axes for axis in axes)
        reduced_axes_set = set(axis for axes in reduced_axes for axis in axes)
        batches_axes = tuple(axis for axis, space in enumerate(self.spaces)
                             if axis not in measure_axes_set and axis not in reduced_axes_set)

        results, prob, state_value = self.backend.measure_mixed_state(
            self.values(), batches_axes, reduced_axes, measure_axes, results)

        results = results[..., 0] if single else results
        state = MixedStateTensor.of(state_value, [
            *(self.spaces[axis] for axis in batches_axes),
            *spaces,
            *(space.ct for space in spaces),
            *(self.spaces[axes[0]] for axes in reduced_axes),
            *(self.spaces[axes[1]] for axes in reduced_axes),
        ], backend=self.backend)

        return results, prob, state


class OperatorTensor(QTensor[BackendValue]):
    @classmethod
    def of(cls,
        values: Union[QTensor, Any],
        spaces: Optional[Iterable[Space]] = None,
    ) -> 'OperatorTensor':
        if isinstance(values, OperatorTensor):
            return values
        if isinstance(values, QTensor):
            # noinspection PyTypeChecker
            return cls(values._values, values._spaces)
        return cls(values, spaces)

    @classmethod
    def from_matrix(cls,
        matrix: BackendValue,
        ket_spaces: Union[Iterable[KetSpace], KetSpace],
        num_spaces: Union[Iterable[NumSpace], NumSpace] = (),
    ) -> 'OperatorTensor':
        num_spaces = (num_spaces,) if isinstance(num_spaces, NumSpace) else tuple(num_spaces)
        ket_spaces = (ket_spaces,) if isinstance(ket_spaces, KetSpace) else tuple(ket_spaces)
        bra_spaces = tuple(space.ct for space in ket_spaces)
        spaces = num_spaces + ket_spaces + bra_spaces
        shape = tuple(space.n for space in spaces)
        values = get_default_backend().reshape(matrix, shape)
        return cls(values, spaces)

    def __init__(self, values: Any, spaces: Iterable[Space]):
        spaces = tuple(spaces)
        ket_spaces = _match_spaces_pairs(spaces)
        super().__init__(values, spaces)
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
