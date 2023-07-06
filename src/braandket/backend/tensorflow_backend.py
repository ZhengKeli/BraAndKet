from typing import Any, Iterable, Optional, Union

import tensorflow as tf

from .backend import Backend


class TensorflowBackend(Backend[tf.Tensor]):

    # basics

    def convert(self, values: Any, *, dtype=None) -> tf.Tensor:
        values = tf.convert_to_tensor(values)
        if dtype is not None:
            values = tf.cast(values, dtype=dtype)
        elif values.dtype in (tf.int8, tf.int16, tf.int32, tf.int64):
            values = tf.cast(values, dtype=tf.int32)
        elif values.dtype in (tf.float16, tf.float32, tf.float64):
            values = tf.cast(values, dtype=tf.float32)
        elif values.dtype in (tf.complex64, tf.complex128):
            values = tf.cast(values, dtype=tf.complex64)
        else:
            raise TypeError(f"Unsupported dtype of values: {values.dtype}")
        return values

    def copy(self, values: tf.Tensor) -> tf.Tensor:
        return values

    def _auto_cast(self, values0: tf.Tensor, values1: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        ordered_dtypes = [tf.complex64, tf.float32, tf.int32]
        dt0 = ordered_dtypes.index(values0.dtype)
        dt1 = ordered_dtypes.index(values1.dtype)
        dtype = ordered_dtypes[dt0] if dt0 < dt1 else ordered_dtypes[dt1]
        values0 = tf.cast(values0, dtype=dtype)
        values1 = tf.cast(values1, dtype=dtype)
        return values0, values1

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

    def pow(self, values0: tf.Tensor, values1: tf.Tensor) -> tf.Tensor:
        return tf.pow(values0, values1)

    def square(self, values: tf.Tensor) -> tf.Tensor:
        return tf.square(values)

    def sqrt(self, values: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(values)

    def exp(self, values: tf.Tensor) -> tf.Tensor:
        return tf.exp(values)

    def sin(self, values: tf.Tensor) -> tf.Tensor:
        return tf.sin(values)

    def cos(self, values: tf.Tensor) -> tf.Tensor:
        return tf.cos(values)

    def conj(self, values: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(values)

    def abs(self, values: tf.Tensor) -> tf.Tensor:
        return tf.abs(values)

    # linear operations

    def add(self, values0: tf.Tensor, values1: tf.Tensor) -> tf.Tensor:
        values0, values1 = self._auto_cast(values0, values1)
        return values0 + values1

    def sub(self, values0: tf.Tensor, values1: tf.Tensor) -> tf.Tensor:
        values0, values1 = self._auto_cast(values0, values1)
        return values0 - values1

    def mul(self, values0: tf.Tensor, values1: tf.Tensor) -> tf.Tensor:
        values0, values1 = self._auto_cast(values0, values1)
        return values0 * values1

    def div(self, values0: tf.Tensor, values1: tf.Tensor) -> tf.Tensor:
        values0, values1 = self._auto_cast(values0, values1)
        return values0 / values1

    # operator operations

    def ensure_shape(self, values: tf.Tensor, shape: Iterable[int]) -> tf.Tensor:
        return tf.ensure_shape(values, shape)

    def reshape(self, values: tf.Tensor, shape: Iterable[int]) -> tf.Tensor:
        return tf.reshape(values, shape)

    def transpose(self, values: tf.Tensor, *, axes: Iterable[int]) -> tf.Tensor:
        return tf.transpose(values, axes)

    def expand(self, values: tf.Tensor, axes: Iterable[int], sizes: Optional[Iterable[int]] = None) -> tf.Tensor:
        for axis in axes:
            values = tf.expand_dims(values, axis)
        if sizes is not None:
            sizes = tuple(sizes)
            for axis, size in zip(axes, sizes, strict=True):
                values = tf.repeat(values, size, axis)
        return values

    def slice(self, values: tf.Tensor, *, slices: Union[int, slice, Iterable[Union[int, slice]]]) -> tf.Tensor:
        return values[slices]

    def trace(self, values: tf.Tensor, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            values = tf.linalg.trace(values, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return values

    def diag(self, values: tf.Tensor, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        axis_pairs = tf.transpose(axes)  # [axes_n, 2]
        while len(axes) > 0:
            axis0, axis1 = axis_pairs[0]
            values = tf.linalg.diag(values, axis0, axis1)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis0, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
        return values

    def dot(self,
        values0: tf.Tensor, values1: tf.Tensor, *,
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
            raise ValueError("Found duplication for axes of values0 !")
        rem_axes0 = tuple(axis for axis in range(ndim0) if axis not in selected_axes0)

        selected_axes1 = {*bat_axes1, *dot_axes1}
        if len(selected_axes1) != len(bat_axes1) + len(dot_axes1):
            raise ValueError("Found duplication for axes of values1 !")
        rem_axes1 = tuple(axis for axis in range(ndim1) if axis not in selected_axes1)

        values0 = tf.transpose(values0, [*bat_axes0, *rem_axes0, *dot_axes0])
        values1 = tf.transpose(values1, [*bat_axes1, *rem_axes1, *dot_axes1])

        bat_axes_n, dot_axes_n = len(bat_axes0), len(dot_axes0)
        rem_axes0_n, rem_axes1_n = len(rem_axes0), len(rem_axes1)

        values0 = self.expand(values0, tuple((i + bat_axes_n + rem_axes0_n) for i in range(rem_axes1_n)))
        values1 = self.expand(values1, tuple((i + bat_axes_n) for i in range(rem_axes0_n)))
        # [*bat_axes, *rem_axes0, *exp_axes0, *dot_axes]
        # [*bat_axes, *exp_axes1, *rem_axes1, *dot_axes]

        values = self.mul(values0, values1)
        # [*bat_axes, *rem_axes0, *rem_axes1, *dot_axes]

        values = tf.reduce_sum(values, tuple((i + bat_axes_n + rem_axes0_n + rem_axes1_n) for i in range(dot_axes_n)))
        # [*bat_axes, *rem_axes0, *rem_axes1]

        return values, (rem_axes0, rem_axes1)

    # special

    def take(self, values: Iterable[tf.Tensor], indices: tf.Tensor) -> tf.Tensor:
        values = tf.stack(values, axis=-1)
        indices = tf.expand_dims(indices, axis=-1)
        indices = tf.cast(indices, tf.int32)
        values = tf.experimental.numpy.take_along_axis(values, indices, axis=-1)
        values = tf.squeeze(values, axis=-1)
        return values

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
