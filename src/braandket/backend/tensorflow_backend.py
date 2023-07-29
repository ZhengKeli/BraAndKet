from typing import Iterable, Optional, Union

import numpy as np
import tensorflow as tf

from .backend import ArrayLike, Backend


class TensorflowBackend(Backend[tf.Tensor]):

    # basics

    def convert(self, value: ArrayLike, *values: ArrayLike) -> Union[tf.Tensor, tuple[tf.Tensor, ...]]:
        value = tf.convert_to_tensor(value)
        if not values:
            return value

        values = tuple(tf.convert_to_tensor(v) for v in values)
        dtype = get_compact_dtype(value.dtype, *(v.dtype for v in values))

        value = tf.cast(value, dtype)
        values = tuple(tf.cast(v, dtype) for v in values)
        return value, *values

    def copy(self, values: ArrayLike) -> tf.Tensor:
        return self.convert(values)

    # constructors

    def zeros(self, shape: Iterable[int], *, dtype=tf.float32) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def ones(self, shape: Iterable[int], *, dtype=tf.float32) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def onehot(self, index: int, size: int, *, dtype=tf.float32) -> tf.Tensor:
        one_value = tf.ones((), dtype=dtype)
        zero_value = tf.zeros((), dtype=dtype)
        return tf.one_hot(index, size, on_value=one_value, off_value=zero_value)

    def eye(self, size: int, *, dtype=tf.float32) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    # unary operations

    def pow(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.convert(value0, value1)
        return tf.pow(value0, value1)

    def square(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.square(value)

    def sqrt(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.sqrt(value)

    def exp(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.exp(value)

    def sin(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.sin(value)

    def cos(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.cos(value)

    def conj(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.math.conj(value)

    def abs(self, value: ArrayLike) -> tf.Tensor:
        value = self.convert(value)
        return tf.abs(value)

    # linear operations

    def add(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.convert(value0, value1)
        return value0 + value1

    def sub(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.convert(value0, value1)
        return value0 - value1

    def mul(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.convert(value0, value1)
        return value0 * value1

    def div(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.convert(value0, value1)
        return value0 / value1

    # operator operations

    def ensure_shape(self, value: ArrayLike, shape: Iterable[int]) -> tf.Tensor:
        value = self.convert(value)
        return tf.ensure_shape(value, shape)

    def reshape(self, value: ArrayLike, shape: Iterable[int]) -> tf.Tensor:
        value = self.convert(value)
        return tf.reshape(value, shape)

    def transpose(self, value: ArrayLike, *, axes: Iterable[int]) -> tf.Tensor:
        value = self.convert(value)
        return tf.transpose(value, axes)

    def expand(self, value: ArrayLike, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> tf.Tensor:
        value = self.convert(value)
        for axis in axes:
            value = tf.expand_dims(value, axis)
        if sizes is not None:
            sizes = tuple(sizes)
            for axis, size in zip(axes, sizes, strict=True):
                value = tf.repeat(value, size, axis)
        return value

    def slice(self, value: ArrayLike, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> tf.Tensor:
        value = self.convert(value)
        return value[slices]

    def trace(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        value = self.convert(value)
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = tf.linalg.trace(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        value = self.convert(value)
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = tf.linalg.diag(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def dot(self,
        value0: ArrayLike, value1: ArrayLike, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[tf.Tensor, tuple[tuple[int, ...], tuple[int, ...]]]:
        value0, value1 = self.convert(value0, value1)

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

        value0 = tf.transpose(value0, [*bat_axes0, *rem_axes0, *dot_axes0])
        value1 = tf.transpose(value1, [*bat_axes1, *rem_axes1, *dot_axes1])

        bat_axes_n, dot_axes_n = len(bat_axes0), len(dot_axes0)
        rem_axes0_n, rem_axes1_n = len(rem_axes0), len(rem_axes1)

        value0 = self.expand(value0, tuple((i + bat_axes_n + rem_axes0_n) for i in range(rem_axes1_n)))
        value1 = self.expand(value1, tuple((i + bat_axes_n) for i in range(rem_axes0_n)))
        # [*bat_axes, *rem_axes0, *exp_axes0, *dot_axes]
        # [*bat_axes, *exp_axes1, *rem_axes1, *dot_axes]

        value = self.mul(value0, value1)
        # [*bat_axes, *rem_axes0, *rem_axes1, *dot_axes]

        value = tf.reduce_sum(value, tuple((i + bat_axes_n + rem_axes0_n + rem_axes1_n) for i in range(dot_axes_n)))
        # [*bat_axes, *rem_axes0, *rem_axes1]

        return value, (rem_axes0, rem_axes1)

    # special

    def take(self, values: Iterable[ArrayLike], indices: ArrayLike) -> tf.Tensor:
        values = self.convert(values) if isinstance(values, (np.ndarray, tf.Tensor)) else self.convert(*values)
        values = tf.stack(values, axis=-1)
        indices = tf.expand_dims(indices, axis=-1)
        indices = tf.cast(indices, tf.int32)
        value = tf.experimental.numpy.take_along_axis(values, indices, axis=-1)
        value = tf.squeeze(value, axis=-1)
        return value

    def choose(self, probs: Iterable[ArrayLike]) -> tf.Tensor:
        probs = self.convert(probs) if isinstance(probs, (np.ndarray, tf.Tensor)) else self.convert(*probs)
        probs = tf.stack(probs, axis=-1)  # [*batch_shape, choose_n]
        batch_shape = tf.shape(probs)[:-1]
        batch_size = tf.reduce_prod(batch_shape)
        probs = tf.reshape(probs, [batch_size, -1])  # [batch_size, choose_n]
        logits = tf.math.abs(tf.math.log(probs))

        choice = tf.random.categorical(logits, 1)  # [batch_size, 1]
        choice = choice[:, 0]  # [batch_size]
        choice = tf.reshape(choice, batch_shape)  # [*batch_shape]
        return choice


tensorflow_backend = TensorflowBackend()

# utils

_grouped_dtypes = (
    (tf.bool,),
    (None, tf.int8, tf.int16, tf.int32, tf.int64),
    (None, None, tf.float16, tf.float32, tf.float64),
    (None, None, None, tf.complex64, tf.complex128))

_supported_dtypes = tuple(dt for gp in _grouped_dtypes for dt in gp if dt is not None)


def get_dtype_indices(dtype: tf.DType) -> tuple[int, int]:
    for group_i, group in enumerate(_grouped_dtypes):
        for item_i, dt in enumerate(group):
            if dtype == dt:
                return group_i, item_i
    raise TypeError(f"Unsupported dtype {dtype}! Supported dtypes {','.join(map(repr, _supported_dtypes))}")


def get_compact_dtype(dtype0: tf.DType, *dtypes: tf.DType) -> tf.DType:
    group_i, item_i = get_dtype_indices(dtype0)
    for dt in dtypes:
        gi, ii = get_dtype_indices(dt)
        group_i = max(group_i, gi)
        item_i = max(item_i, ii)
    return _grouped_dtypes[group_i][item_i]
