from typing import Any, Iterable, Optional, Union

import tensorflow as tf

from .backend import Backend


class TensorflowBackend(Backend[tf.Tensor]):

    # basics

    def convert(self, value: Any, *, dtype=None) -> tf.Tensor:
        value = tf.convert_to_tensor(value)
        if dtype is not None:
            value = tf.cast(value, dtype=dtype)
        elif value.dtype in (tf.int8, tf.int16, tf.int32, tf.int64):
            value = tf.cast(value, dtype=tf.int32)
        elif value.dtype in (tf.float16, tf.float32, tf.float64):
            value = tf.cast(value, dtype=tf.float32)
        elif value.dtype in (tf.complex64, tf.complex128):
            value = tf.cast(value, dtype=tf.complex64)
        else:
            raise TypeError(f"Unsupported dtype of value: {value.dtype}")
        return value

    def copy(self, value: tf.Tensor) -> tf.Tensor:
        return value

    def _auto_cast(self, value0: tf.Tensor, value1: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        ordered_dtypes = [tf.complex64, tf.float32, tf.int32]
        dt0 = ordered_dtypes.index(value0.dtype)
        dt1 = ordered_dtypes.index(value1.dtype)
        dtype = ordered_dtypes[dt0] if dt0 < dt1 else ordered_dtypes[dt1]
        value0 = tf.cast(value0, dtype=dtype)
        value1 = tf.cast(value1, dtype=dtype)
        return value0, value1

    # constructors

    def zeros(self, shape: Iterable[int], *, dtype=tf.complex64) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def ones(self, shape: Iterable[int], *, dtype=tf.complex64) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def onehot(self, index: int, size: int, *, dtype=tf.complex64) -> tf.Tensor:
        one_value = tf.ones((), dtype=dtype)
        zero_value = tf.zeros((), dtype=dtype)
        return tf.one_hot(index, size, on_value=one_value, off_value=zero_value)

    def eye(self, size: int, *, dtype=tf.complex64) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    # unary operations

    def pow(self, value0: tf.Tensor, value1: tf.Tensor) -> tf.Tensor:
        return tf.pow(value0, value1)

    def square(self, value: tf.Tensor) -> tf.Tensor:
        return tf.square(value)

    def sqrt(self, value: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(value)

    def exp(self, value: tf.Tensor) -> tf.Tensor:
        return tf.exp(value)

    def sin(self, value: tf.Tensor) -> tf.Tensor:
        return tf.sin(value)

    def cos(self, value: tf.Tensor) -> tf.Tensor:
        return tf.cos(value)

    def conj(self, value: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(value)

    def abs(self, value: tf.Tensor) -> tf.Tensor:
        return tf.abs(value)

    # linear operations

    def add(self, value0: tf.Tensor, value1: tf.Tensor) -> tf.Tensor:
        value0, value1 = self._auto_cast(value0, value1)
        return value0 + value1

    def sub(self, value0: tf.Tensor, value1: tf.Tensor) -> tf.Tensor:
        value0, value1 = self._auto_cast(value0, value1)
        return value0 - value1

    def mul(self, value0: tf.Tensor, value1: tf.Tensor) -> tf.Tensor:
        value0, value1 = self._auto_cast(value0, value1)
        return value0 * value1

    def div(self, value0: tf.Tensor, value1: tf.Tensor) -> tf.Tensor:
        value0, value1 = self._auto_cast(value0, value1)
        return value0 / value1

    # operator operations

    def ensure_shape(self, value: tf.Tensor, shape: Iterable[int]) -> tf.Tensor:
        return tf.ensure_shape(value, shape)

    def reshape(self, value: tf.Tensor, shape: Iterable[int]) -> tf.Tensor:
        return tf.reshape(value, shape)

    def transpose(self, value: tf.Tensor, *, axes: Iterable[int]) -> tf.Tensor:
        return tf.transpose(value, axes)

    def expand(self, value: tf.Tensor, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> tf.Tensor:
        for axis in axes:
            value = tf.expand_dims(value, axis)
        if sizes is not None:
            sizes = tuple(sizes)
            for axis, size in zip(axes, sizes, strict=True):
                value = tf.repeat(value, size, axis)
        return value

    def slice(self, value: tf.Tensor, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> tf.Tensor:
        return value[slices]

    def trace(self, value: tf.Tensor, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = tf.linalg.trace(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def diag(self, value: tf.Tensor, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            value = tf.linalg.diag(value, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return value

    def dot(self,
        value0: tf.Tensor, value1: tf.Tensor, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[tf.Tensor, tuple[tuple[int, ...], tuple[int, ...]]]:
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

    def take(self, values: Iterable[tf.Tensor], indices: tf.Tensor) -> tf.Tensor:
        values = tf.stack(values, axis=-1)
        indices = tf.expand_dims(indices, axis=-1)
        indices = tf.cast(indices, tf.int32)
        value = tf.experimental.numpy.take_along_axis(values, indices, axis=-1)
        value = tf.squeeze(value, axis=-1)
        return value

    def choose(self, probs: Iterable[tf.Tensor]) -> tf.Tensor:
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
