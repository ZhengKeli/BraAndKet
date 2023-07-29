from typing import Iterable, Optional, Union

import numpy as np

from .backend import ArrayLike, Backend


class NumpyBackend(Backend[np.ndarray]):

    # basics

    def convert(self, value: ArrayLike, *, dtype=None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    def copy(self, value: ArrayLike) -> np.ndarray:
        return np.copy(value)

    # constructors

    def zeros(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def onehot(self, index: int, size: int, *, dtype=np.float32) -> np.ndarray:
        value = np.zeros(size, dtype=dtype)
        value[index] = 1.0
        return value

    def eye(self, size: int, *, dtype=np.float32) -> np.ndarray:
        return np.eye(size, dtype=dtype)

    # unary operations

    def pow(self, value0: ArrayLike, value1: ArrayLike) -> np.ndarray:
        return np.power(value0, value1)

    def square(self, value: ArrayLike) -> np.ndarray:
        return np.square(value)

    def sqrt(self, value: ArrayLike) -> np.ndarray:
        return np.sqrt(value)

    def exp(self, value: ArrayLike) -> np.ndarray:
        return np.exp(value)

    def sin(self, value: ArrayLike) -> np.ndarray:
        return np.sin(value)

    def cos(self, value: ArrayLike) -> np.ndarray:
        return np.cos(value)

    def conj(self, value: ArrayLike) -> np.ndarray:
        return np.conj(value)

    def abs(self, value: ArrayLike) -> np.ndarray:
        return np.abs(value)

    # linear operations

    def add(self, value0: ArrayLike, value1: ArrayLike) -> np.ndarray:
        return value0 + value1

    def sub(self, value0: ArrayLike, value1: ArrayLike) -> np.ndarray:
        return value0 - value1

    def mul(self, value0: ArrayLike, value1: ArrayLike) -> np.ndarray:
        return value0 * value1

    def div(self, value0: ArrayLike, value1: ArrayLike) -> np.ndarray:
        return value0 / value1

    # tensor operations

    def ensure_shape(self, value: ArrayLike, shape: Iterable[int]):
        if np.shape(value) != tuple(shape):
            raise ValueError(f"Unexpected value shape! expected={shape}, actual={np.shape(value)}")
        return value

    def reshape(self, value: ArrayLike, shape: Iterable[int]) -> np.ndarray:
        return np.reshape(value, shape)

    def transpose(self, value: ArrayLike, *, axes: Iterable[int]) -> np.ndarray:
        return np.transpose(value, axes)

    def expand(self, value: ArrayLike, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> np.ndarray:
        axes = tuple(axes)
        value = np.expand_dims(value, axes)
        if sizes is not None:
            sizes = tuple(sizes)
            for axis, size in zip(axes, sizes, strict=True):
                value = np.repeat(value, size, axis)
        return value

    def slice(self, value: ArrayLike, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> np.ndarray:
        return value[slices]

    def trace(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> np.ndarray:
        axis_pairs = np.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = np.trace(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> np.ndarray:
        axis_pairs = np.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = np.diagonal(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def dot(self,
        value0: ArrayLike, value1: ArrayLike, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[np.ndarray, tuple[tuple[int, ...], tuple[int, ...]]]:
        bat_axes0, bat_axes1 = tuple(bat_axes[0]), tuple(bat_axes[1])
        if not len(bat_axes0) == len(bat_axes1):
            raise ValueError("len(bat_axes[0]) != len(bat_axes[1])")
        del bat_axes

        dot_axes0, dot_axes1 = tuple(dot_axes[0]), tuple(dot_axes[1])
        if not len(dot_axes0) == len(dot_axes1):
            raise ValueError("len(dot_axes[0]) != len(dot_axes[1])")
        del dot_axes

        selected_axes0 = {*bat_axes0, *dot_axes0}
        if len(selected_axes0) != len(bat_axes0) + len(dot_axes0):
            raise ValueError("Found duplication for axes of value0 !")
        rem_axes0 = tuple(axis for axis in range(ndim0) if axis not in selected_axes0)

        selected_axes1 = {*bat_axes1, *dot_axes1}
        if len(selected_axes1) != len(bat_axes1) + len(dot_axes1):
            raise ValueError("Found duplication for axes of value1 !")
        rem_axes1 = tuple(axis for axis in range(ndim1) if axis not in selected_axes1)

        value0 = np.transpose(value0, [*bat_axes0, *rem_axes0, *dot_axes0])
        value1 = np.transpose(value1, [*bat_axes1, *rem_axes1, *dot_axes1])
        # [*bat_axes, *rem_axes, *dot_axes]

        bat_axes_n, dot_axes_n = len(bat_axes0), len(dot_axes0)
        rem_axes0_n, rem_axes1_n = len(rem_axes0), len(rem_axes1)

        value0 = np.expand_dims(value0, axis=tuple(bat_axes_n + rem_axes0_n + np.arange(rem_axes1_n)))
        value1 = np.expand_dims(value1, axis=tuple(bat_axes_n + np.arange(rem_axes0_n)))
        # [*bat_axes, *rem_axes0, *exp_axes0, *dot_axes]
        # [*bat_axes, *exp_axes1, *rem_axes1, *dot_axes]

        value = value0 * value1
        # [*bat_axes, *rem_axes0, *rem_axes1, *dot_axes]

        value = np.sum(value, axis=tuple(bat_axes_n + rem_axes0_n + rem_axes1_n + np.arange(dot_axes_n)))
        # [*bat_axes, *rem_axes0, *rem_axes1]

        return value, (rem_axes0, rem_axes1)

    # special

    def take(self, values: Iterable[ArrayLike], indices: ArrayLike) -> np.ndarray:
        values = np.stack(values, axis=-1)
        indices = np.expand_dims(indices, axis=-1)
        value = np.take_along_axis(values, indices, axis=-1)
        value = np.squeeze(value, axis=-1)
        return value

    def choose(self, probs: Iterable[ArrayLike]) -> tuple[np.ndarray, np.ndarray]:
        probs = np.stack(probs, axis=-1)  # [batch_size, choose_n]
        probs /= np.sum(probs, axis=-1, keepdims=True)
        probs = np.abs(probs)
        choice = np.apply_along_axis(lambda p: np.random.choice(len(p), p=p), axis=-1, arr=probs)  # [batch_size]
        return choice


numpy_backend = NumpyBackend()
