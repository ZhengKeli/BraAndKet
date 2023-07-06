import abc
from typing import Any, Generic, Iterable, Optional, Union

from braandket.backend import Backend, ValuesType, get_default_backend
from braandket.space import BraSpace, HSpace, KetSpace, NumSpace, Space
from braandket.utils import iter_structure


class QTensor(Generic[ValuesType], abc.ABC):

    @classmethod
    def of(cls, values: Any, spaces: Iterable[Space], *, backend: Optional[Backend] = None) -> 'QTensor':
        if backend is None:
            backend = get_default_backend()
        spaces = tuple(spaces)

        try:
            from braandket import NumericTensor
            # noinspection PyTypeChecker
            return NumericTensor(values, spaces, backend)
        except TypeError:
            pass

        try:
            from .special import PureStateTensor
            # noinspection PyTypeChecker
            return PureStateTensor(values, spaces, backend)
        except TypeError:
            pass

        return QTensor(values, spaces, backend)

    def __init__(self, values: Any, spaces: Iterable[Space], backend: Backend):
        # check spaces
        spaces = tuple(spaces)
        for i in range(len(spaces)):
            for j in range(i + 1, len(spaces)):
                if spaces[i] == spaces[j]:
                    raise ValueError(f"Found duplicated spaces when constructing QTensor: {spaces[i]}")

        # check values
        values = backend.convert(values)
        values = backend.ensure_shape(values, tuple(space.n for space in spaces))

        # construct
        self._spaces = spaces
        self._values = values
        self._backend = backend

    @property
    def backend(self) -> Backend[ValuesType]:
        return self._backend

    def spawn(self, values: Any, spaces: Iterable[Space]) -> 'QTensor':
        """ create a new BackendTensor instance with the same backend """
        return QTensor.of(values, spaces, backend=self.backend)

    # basic operations

    @property
    def spaces(self) -> tuple[Space, ...]:
        return self._spaces

    def values(self, *slices: Union[Space, tuple[Space, Union[int, None]]]) -> ValuesType:
        """ get values from this tensor """
        return self.values_and_slices(*slices)[0]

    def values_and_slices(self,
        *slices: Union[Space, tuple[Space, Union[int, None]]]
    ) -> tuple[ValuesType, tuple[Union[Space, tuple[Space, int]], ...]]:
        if len(slices) == 0:
            return self._values, self._spaces

        # parse arguments
        sp_slices = slices
        spaces = []
        slices = []
        for sl in sp_slices:
            if isinstance(sl, Space):
                space, index = sl, None
            else:
                space, index = sl
            spaces.append(space)
            slices.append(index)

        # find axes of the specified spaces
        axes = []
        for space in spaces:
            axis = self._spaces.index(space)
            axes.append(axis)

        # fill axes of unspecified spaces
        axes_set = frozenset(axes)
        for axis, space in enumerate(self._spaces):
            if axis not in axes_set:
                axes.append(axis)
                spaces.append(space)
                slices.append(None)

        # values slices
        values_slices = tuple((slice(None) if sl is None else sl) for sl in slices)

        # manipulate values
        values = self._values
        values = self.backend.transpose(values, axes=axes)
        values = self.backend.slice(values, slices=values_slices)

        # slices for output
        sp_slices = tuple((sp if sl is None else (sp, sl)) for sp, sl in zip(spaces, slices))

        return values, sp_slices

    # common operations

    def __getitem__(self, items: Union[tuple[tuple[Space, int], ...], dict[Space, int]]) -> 'QTensor':
        if not isinstance(items, dict):
            items = dict(items)

        slices = tuple(items.get(sp, slice(None)) for sp in self._spaces)
        new_spaces = tuple(sp for sp in self._spaces if sp not in items)
        new_values = self.backend.slice(self._values, slices=slices)
        return self.spawn(new_values, new_spaces)

    def __iter__(self):
        raise TypeError("QTensor is not allowed to be iterated.")

    def __eq__(self, other):
        if self.is_scalar:
            return self.scalar() == other
        raise TypeError("The function __eq__() of QTensor not allowed, because it is ambiguous.")

    def __copy__(self) -> 'QTensor':
        return self.spawn(self.backend.copy(self._values), self._spaces)

    def copy(self) -> 'QTensor':
        return self.__copy__()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} spaces={repr(self._spaces)}, values={repr(self._values)}>"

    def __str__(self) -> str:
        return repr(self)

    # scalar operations

    @property
    def is_scalar(self) -> bool:
        return len(self.spaces) == 0

    def scalar(self) -> ValuesType:
        """ get a scalar value from zero-dimension tensor """
        if not self.is_scalar:
            raise ValueError(f"This QTensor is not a scalar. It has {len(self.spaces)} spaces.")
        return self.values()

    def __float__(self) -> float:
        return float(self.scalar())

    # linear operations

    def __pos__(self) -> 'QTensor':
        return self

    def __neg__(self) -> 'QTensor':
        return (-1) * self

    def __add__(self, other) -> 'QTensor':
        if not isinstance(other, QTensor):
            other = self.spawn(other, ())

        self_expanded, other_expanded = self._expand_tensors_for_addsub(self, other)

        new_spaces = tuple(self_expanded._spaces)
        self_values = self_expanded._values
        other_values = other_expanded.values(*new_spaces)
        new_values = self_expanded.backend.add(self_values, other_values)
        return self_expanded.spawn(new_values, new_spaces)

    def __radd__(self, other) -> 'QTensor':
        return self + other

    def __sub__(self, other) -> 'QTensor':
        return self + (-other)

    def __rsub__(self, other) -> 'QTensor':
        return -self + other

    @classmethod
    def _expand_tensors_for_addsub(cls, tensor0: 'QTensor', tensor1: 'QTensor') -> tuple['QTensor', 'QTensor']:
        from .operations import _broadcast_num_spaces
        tensor0, tensor1 = _broadcast_num_spaces(tensor0, tensor1)
        return tensor0, tensor1

    def __mul__(self, other) -> 'QTensor':
        if not isinstance(other, QTensor):
            other = self.spawn(other, ())

        self_expanded, other_expanded = self._expand_tensors_for_muldiv(self, other)

        new_spaces = self_expanded._spaces
        self_values = self_expanded._values
        other_values = other_expanded.values(*new_spaces)
        new_values = self_expanded.backend.mul(self_values, other_values)
        return self_expanded.spawn(new_values, new_spaces)

    def __rmul__(self, other) -> 'QTensor':
        return self * other

    def __truediv__(self, other) -> 'QTensor':
        if not isinstance(other, QTensor):
            other = self.spawn(other, ())

        self_expanded, other_expanded = self._expand_tensors_for_muldiv(self, other)

        new_spaces = self_expanded._spaces
        self_values = self_expanded._values
        other_values = other_expanded.values(*new_spaces)
        new_values = self_expanded.backend.div(self_values, other_values)
        return self_expanded.spawn(new_values, new_spaces)

    def __rtruediv__(self, other) -> 'QTensor':
        other = self.spawn(other, ())
        return other / self

    @classmethod
    def _expand_tensors_for_muldiv(cls, tensor0: 'QTensor', tensor1: 'QTensor') -> tuple['QTensor', 'QTensor']:
        from .operations import _broadcast_h_spaces, _broadcast_num_spaces
        tensor0, tensor1 = _broadcast_h_spaces(tensor0, tensor1)
        tensor0, tensor1 = _broadcast_num_spaces(tensor0, tensor1)
        return tensor0, tensor1

    # tensor operations

    def __matmul__(self, other) -> 'QTensor':
        if not isinstance(other, QTensor):
            return self * other

        self_expanded, other_expanded = self._expand_tensors_for_matmul(self, other)

        self_spaces = self_expanded._spaces
        self_values = self_expanded._values
        other_spaces = other_expanded._spaces
        other_values = other_expanded._values

        self_dot_axes, other_dot_axes = [], []
        self_num_axes, other_num_axes = [], []
        for self_axis, self_space in enumerate(self_spaces):
            if isinstance(self_space, KetSpace):
                pass  # ignored
            elif isinstance(self_space, BraSpace):
                try:
                    other_axis = other_spaces.index(self_space.ct)
                    self_dot_axes.append(self_axis)
                    other_dot_axes.append(other_axis)
                except ValueError:
                    pass  # ignored
            elif isinstance(self_space, NumSpace):
                other_axis = other_spaces.index(self_space)
                self_num_axes.append(self_axis)
                other_num_axes.append(other_axis)

        new_values, (self_rem_axes, other_rem_axes) = self_expanded.backend.dot(
            self_values, other_values,
            ndim0=len(self_spaces), ndim1=len(other_spaces),
            dot_axes=(self_dot_axes, other_dot_axes),
            bat_axes=(self_num_axes, other_num_axes))
        new_spaces = [
            *(self_spaces[axis] for axis in self_num_axes),
            *(self_spaces[axis] for axis in self_rem_axes),
            *(other_spaces[axis] for axis in other_rem_axes)]
        return self_expanded.spawn(new_values, new_spaces)

    def __rmatmul__(self, other) -> 'QTensor':
        return self @ other  # (other is scalar)

    @classmethod
    def _expand_tensors_for_matmul(cls, tensor0: 'QTensor', tensor1: 'QTensor') -> tuple['QTensor', 'QTensor']:
        from .operations import _broadcast_num_spaces
        # noinspection PyTypeChecker
        return _broadcast_num_spaces(tensor0, tensor1)

    # spaces operations

    @property
    def ct(self) -> 'QTensor':
        new_spaces = tuple((space.ct if isinstance(space, HSpace) else space) for space in self._spaces)
        new_values = self.backend.conj(self._values)
        return self.spawn(new_values, new_spaces)

    # specialize

    def as_numeric_tensor(self):
        from braandket import NumericTensor
        return NumericTensor.of(self)

    def as_pure_state_tensor(self):
        from .special import PureStateTensor
        return PureStateTensor.of(self)

    def as_mixed_state_tensor(self):
        from .special import MixedStateTensor
        return MixedStateTensor.of(self)

    def as_operator_tensor(self):
        from .special import OperatorTensor
        return OperatorTensor.of(self)

    # inflate & flatten

    @classmethod
    def inflate(cls,
        values: ValuesType,
        spaces: Iterable[Union[NumSpace, KetSpace, BraSpace, Iterable[Union[NumSpace, KetSpace, BraSpace]]]], *,
        backend: Optional[Backend] = None
    ) -> 'QTensor':
        if backend is None:
            backend = get_default_backend()
        spaces = tuple(iter_structure(spaces))
        shape = tuple(space.n for space in spaces)
        values = backend.reshape(values, shape)
        return cls.of(values, spaces, backend=backend)

    def flatten(self,
        spaces: Optional[Iterable[Union[NumSpace, KetSpace, BraSpace]]] = None, *,
        num_spaces: Optional[Iterable[NumSpace]] = None,
        ket_spaces: Optional[Iterable[KetSpace]] = None,
        bra_spaces: Optional[Iterable[BraSpace]] = None,
    ) -> tuple[ValuesType, tuple[Union[NumSpace, tuple[KetSpace, ...], tuple[BraSpace, ...]], ...]]:
        num_spaces_sl = tuple(space for space in self.spaces if isinstance(space, NumSpace))
        ket_spaces_sl = tuple(space for space in self.spaces if isinstance(space, KetSpace))
        bra_spaces_sl = tuple(space for space in self.spaces if isinstance(space, BraSpace))

        spaces = tuple(spaces) if spaces is not None else ()
        num_spaces_sp = tuple(space for space in spaces if isinstance(space, NumSpace))
        ket_spaces_sp = tuple(space for space in spaces if isinstance(space, KetSpace))
        bra_spaces_sp = tuple(space for space in spaces if isinstance(space, BraSpace))

        num_spaces = list(num_spaces) if num_spaces is not None else []
        for space in (*num_spaces_sp, *num_spaces_sl):
            if space not in num_spaces:
                num_spaces.append(space)

        ket_spaces = list(ket_spaces) if ket_spaces is not None else []
        for space in (*ket_spaces_sp, *ket_spaces_sl):
            if space not in ket_spaces:
                ket_spaces.append(space)

        bra_spaces_kt = tuple(space.ct for space in ket_spaces if space.ct in bra_spaces_sl)
        bra_spaces = list(bra_spaces) if bra_spaces is not None else []
        for space in (*bra_spaces_sp, *bra_spaces_kt, *bra_spaces_sl):
            if space not in bra_spaces:
                bra_spaces.append(space)

        shape = (*(space.n for space in num_spaces),
                 prod(*(space.n for space in ket_spaces)),
                 prod(*(space.n for space in bra_spaces)))
        values = self.backend.reshape(self.values(*num_spaces, *ket_spaces, *bra_spaces), shape)
        return values, (*num_spaces, tuple(ket_spaces), tuple(bra_spaces))

    def flattened_values(self,
        spaces: Optional[Iterable[Union[NumSpace, KetSpace, BraSpace]]] = None, *,
        num_spaces: Optional[Iterable[NumSpace]] = None,
        ket_spaces: Optional[Iterable[KetSpace]] = None,
        bra_spaces: Optional[Iterable[BraSpace]] = None,
    ) -> ValuesType:
        return self.flatten(
            spaces,
            num_spaces=num_spaces,
            ket_spaces=ket_spaces,
            bra_spaces=bra_spaces
        )[0]


# utils

def prod(*values: int) -> int:
    prod_value = 1
    for value in values:
        prod_value *= value
    return prod_value
