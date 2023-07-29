import abc
from typing import Any, Generic, Iterable, Optional, TypeVar, Union

import numpy as np

ValuesType = TypeVar('ValuesType')
ArrayLike = Union[ValuesType, np.ndarray, Iterable, bool, int, float, complex, Any]


class Backend(Generic[ValuesType], abc.ABC):

    def __enter__(self):
        from .default import push_context_backend
        push_context_backend(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .default import pop_context_backend
        pop_context_backend()

    # basics

    @abc.abstractmethod
    def convert(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def copy(self, value: ArrayLike) -> ValuesType:
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
    def add(self, value0: ArrayLike, value1: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def sub(self, value0: ArrayLike, value1: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def mul(self, value0: ArrayLike, value1: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def div(self, value0: ArrayLike, value1: ArrayLike) -> ValuesType:
        pass

    # non-linear operations

    @abc.abstractmethod
    def pow(self, value0: ArrayLike, value1: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def square(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def sqrt(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def exp(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def sin(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def cos(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def conj(self, value: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def abs(self, value: ArrayLike) -> ValuesType:
        pass

    # tensor operations

    @abc.abstractmethod
    def ensure_shape(self, value: ArrayLike, shape: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def reshape(self, value: ArrayLike, shape: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def transpose(self, value: ArrayLike, *, axes: Iterable[int]) -> ValuesType:
        pass

    @abc.abstractmethod
    def expand(self, value: ArrayLike, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> ValuesType:
        pass

    @abc.abstractmethod
    def slice(self, value: ArrayLike, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def trace(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> ValuesType:
        pass

    @abc.abstractmethod
    def dot(self,
        value0: ArrayLike, value1: ArrayLike, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[ValuesType, tuple[tuple[int, ...], tuple[int, ...]]]:
        pass

    # special

    @abc.abstractmethod
    def take(self, values: Iterable[ArrayLike], indices: ArrayLike) -> ValuesType:
        pass

    @abc.abstractmethod
    def choose(self, probs: Iterable[ArrayLike]) -> ValuesType:
        pass
