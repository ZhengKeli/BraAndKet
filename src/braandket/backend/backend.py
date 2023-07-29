import abc
from typing import Any, Generic, Iterable, Optional, TypeVar, Union

import numpy as np

BackendValue = TypeVar('BackendValue')
ArrayLike = Union[BackendValue, np.ndarray, Iterable, bool, int, float, complex, Any]


class Backend(Generic[BackendValue], abc.ABC):

    def __enter__(self):
        from .default import push_context_backend
        push_context_backend(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .default import pop_context_backend
        pop_context_backend()

    # basics

    @abc.abstractmethod
    def convert(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def copy(self, value: ArrayLike) -> BackendValue:
        pass

    # constructors

    @abc.abstractmethod
    def zeros(self, shape: Iterable[int]) -> BackendValue:
        pass

    @abc.abstractmethod
    def ones(self, shape: Iterable[int]) -> BackendValue:
        pass

    @abc.abstractmethod
    def onehot(self, index: int, size: int) -> BackendValue:
        pass

    @abc.abstractmethod
    def eye(self, size: int) -> BackendValue:
        pass

    # linear operations

    @abc.abstractmethod
    def add(self, value0: ArrayLike, value1: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def sub(self, value0: ArrayLike, value1: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def mul(self, value0: ArrayLike, value1: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def div(self, value0: ArrayLike, value1: ArrayLike) -> BackendValue:
        pass

    # non-linear operations

    @abc.abstractmethod
    def pow(self, value0: ArrayLike, value1: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def square(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def sqrt(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def exp(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def sin(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def cos(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def conj(self, value: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def abs(self, value: ArrayLike) -> BackendValue:
        pass

    # tensor operations

    @abc.abstractmethod
    def ensure_shape(self, value: ArrayLike, shape: Iterable[int]) -> BackendValue:
        pass

    @abc.abstractmethod
    def reshape(self, value: ArrayLike, shape: Iterable[int]) -> BackendValue:
        pass

    @abc.abstractmethod
    def transpose(self, value: ArrayLike, *, axes: Iterable[int]) -> BackendValue:
        pass

    @abc.abstractmethod
    def expand(self, value: ArrayLike, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> BackendValue:
        pass

    @abc.abstractmethod
    def slice(self, value: ArrayLike, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> BackendValue:
        pass

    @abc.abstractmethod
    def trace(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> BackendValue:
        pass

    @abc.abstractmethod
    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> BackendValue:
        pass

    @abc.abstractmethod
    def dot(self,
        value0: ArrayLike, value1: ArrayLike, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[BackendValue, tuple[tuple[int, ...], tuple[int, ...]]]:
        pass

    # special

    @abc.abstractmethod
    def take(self, values: Iterable[ArrayLike], indices: ArrayLike) -> BackendValue:
        pass

    @abc.abstractmethod
    def choose(self, probs: Iterable[ArrayLike]) -> BackendValue:
        pass
