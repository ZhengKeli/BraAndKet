from typing import Any, Iterable, Optional, Union

import numpy as np

from .backend import Backend, ValuesType


class NumpyBackend(Backend[np.ndarray]):

    # basics

    def convert(self, values: Any, *, dtype=None) -> np.ndarray:
        return np.asarray(values, dtype=dtype)

    def copy(self, values: np.ndarray) -> np.ndarray:
        return np.copy(values)

    # constructors

    def zeros(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def onehot(self, index: int, size: int, *, dtype=np.float32) -> np.ndarray:
        values = np.zeros(size, dtype=dtype)
        values[index] = 1.0
        return values

    def eye(self, size: int, *, dtype=np.float32) -> np.ndarray:
        return np.eye(size, dtype=dtype)

    # unary operations

    def pow(self, values0: np.ndarray, values1: np.ndarray) -> np.ndarray:
        return np.power(values0, values1)

    def square(self, values: np.ndarray) -> np.ndarray:
        return np.square(values)

    def sqrt(self, values: np.ndarray) -> np.ndarray:
        return np.sqrt(values)

    def exp(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values)

    def sin(self, values: np.ndarray) -> np.ndarray:
        return np.sin(values)

    def cos(self, values: np.ndarray) -> np.ndarray:
        return np.cos(values)

    def conj(self, values: np.ndarray) -> np.ndarray:
        return np.conj(values)

    def abs(self, values: np.ndarray) -> np.ndarray:
        return np.abs(values)

    # linear operations

    def add(self, values0: np.ndarray, values1: np.ndarray) -> np.ndarray:
        return values0 + values1

    def sub(self, values0: np.ndarray, values1: np.ndarray) -> np.ndarray:
        return values0 - values1

    def mul(self, values0: np.ndarray, values1: np.ndarray) -> np.ndarray:
        return values0 * values1

    def div(self, values0: np.ndarray, values1: np.ndarray) -> np.ndarray:
        return values0 / values1

    # tensor operations

    def ensure_shape(self, values: np.ndarray, shape: Iterable[int]):
        if np.shape(values) != tuple(shape):
            raise ValueError(f"Unexpected values shape! expected={shape}, actual={np.shape(values)}")
        return values

    def reshape(self, values: np.ndarray, shape: Iterable[int]) -> np.ndarray:
        return np.reshape(values, shape)

    def transpose(self, values: np.ndarray, *, axes: Iterable[int]) -> np.ndarray:
        return np.transpose(values, axes)

    def expand(self, values: np.ndarray, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> np.ndarray:
        axes = tuple(axes)
        values = np.expand_dims(values, axes)
        if sizes is not None:
            sizes = tuple(sizes)
            for axis, size in zip(axes, sizes, strict=True):
                values = np.repeat(values, size, axis)
        return values

    def slice(self, values: np.ndarray, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> np.ndarray:
        return values[slices]

    def trace(self, values: np.ndarray, axes: tuple[Iterable[int], Iterable[int]]) -> np.ndarray:
        axis_pairs = np.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            values = np.trace(values, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return values

    def diag(self, values: np.ndarray, axes: tuple[Iterable[int], Iterable[int]]) -> np.ndarray:
        axis_pairs = np.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            values = np.diagonal(values, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return values

    def dot(self,
        values0: np.ndarray, values1: np.ndarray, *,
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
            raise ValueError("Found duplication for axes of values0 !")
        rem_axes0 = tuple(axis for axis in range(ndim0) if axis not in selected_axes0)

        selected_axes1 = {*bat_axes1, *dot_axes1}
        if len(selected_axes1) != len(bat_axes1) + len(dot_axes1):
            raise ValueError("Found duplication for axes of values1 !")
        rem_axes1 = tuple(axis for axis in range(ndim1) if axis not in selected_axes1)

        values0 = np.transpose(values0, [*bat_axes0, *rem_axes0, *dot_axes0])
        values1 = np.transpose(values1, [*bat_axes1, *rem_axes1, *dot_axes1])
        # [*bat_axes, *rem_axes, *dot_axes]

        bat_axes_n, dot_axes_n = len(bat_axes0), len(dot_axes0)
        rem_axes0_n, rem_axes1_n = len(rem_axes0), len(rem_axes1)

        values0 = np.expand_dims(values0, axis=tuple(bat_axes_n + rem_axes0_n + np.arange(rem_axes1_n)))
        values1 = np.expand_dims(values1, axis=tuple(bat_axes_n + np.arange(rem_axes0_n)))
        # [*bat_axes, *rem_axes0, *exp_axes0, *dot_axes]
        # [*bat_axes, *exp_axes1, *rem_axes1, *dot_axes]

        values = values0 * values1
        # [*bat_axes, *rem_axes0, *rem_axes1, *dot_axes]

        values = np.sum(values, axis=tuple(bat_axes_n + rem_axes0_n + rem_axes1_n + np.arange(dot_axes_n)))
        # [*bat_axes, *rem_axes0, *rem_axes1]

        return values, (rem_axes0, rem_axes1)

    # special

    def take(self, values: Iterable[ValuesType], indices: np.ndarray) -> np.ndarray:
        values = np.stack(values, axis=-1)
        indices = np.expand_dims(indices, axis=-1)
        values = np.take_along_axis(values, indices, axis=-1)
        values = np.squeeze(values, axis=-1)
        return values

    def choose(self, probs: Iterable[ValuesType]) -> tuple[np.ndarray, np.ndarray]:
        probs = np.stack(probs, axis=-1)  # [batch_size, choose_n]
        probs /= np.sum(probs, axis=-1, keepdims=True)
        probs = np.abs(probs)
        choice = np.apply_along_axis(lambda p: np.random.choice(len(p), p=p), axis=-1, arr=probs)  # [batch_size]
        return choice


numpy_backend = NumpyBackend()
