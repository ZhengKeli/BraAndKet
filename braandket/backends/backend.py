import abc
from typing import Any, Generic, Iterable, Optional, Union
from typing import TypeVar


ValuesType = TypeVar('ValuesType')


class Backend(Generic[ValuesType], abc.ABC):

    # basics

    @abc.abstractmethod
    def convert(self, values: Any) -> ValuesType:
        pass

    @abc.abstractmethod
    def copy(self, values: ValuesType) -> ValuesType:
        pass

    # constructors

    @abc.abstractmethod
    def zeros(self, shape: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def ones(self, shape: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def onehot(self, index: int, size: int) -> ValuesType:
        pass

    @abc.abstractmethod
    def eye(self, size: int) -> ValuesType:
        pass

    # linear operations

    @abc.abstractmethod
    def add(self, values0: ValuesType, values1: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def sub(self, values0: ValuesType, values1: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def mul(self, values0: ValuesType, values1: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def div(self, values0: ValuesType, values1: ValuesType) -> ValuesType:
        pass

    # non-linear operations

    @abc.abstractmethod
    def pow(self, values0: ValuesType, values1: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def square(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def sqrt(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def exp(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def sin(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def cos(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def conj(self, values: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def abs(self, values: ValuesType) -> ValuesType:
        pass

    # dim operations

    @abc.abstractmethod
    def transpose(self, values: ValuesType, *, axes: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def expand(self, values: ValuesType, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> ValuesType:
        pass

    @abc.abstractmethod
    def slice(self, values: ValuesType, *, slices: Union[int, slice, tuple[Union[int, slice]]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def trace(self, values: ValuesType, axes: tuple[Iterable[int], Iterable[int]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def diag(self, values: ValuesType, axes: tuple[Iterable[int], Iterable[int]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def dot(self,
            values0: ValuesType, values1: ValuesType, *,
            dot_axes: tuple[Iterable[int], Iterable[int]],
            bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[ValuesType, tuple[tuple[int, ...], tuple[int, ...]]]:
        pass

    # special

    @abc.abstractmethod
    def take(self, values: Iterable[ValuesType], indices: ValuesType) -> ValuesType:
        pass

    @abc.abstractmethod
    def choose(self, probs: Iterable[ValuesType]) -> ValuesType:
        pass
