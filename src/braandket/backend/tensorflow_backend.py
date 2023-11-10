from typing import Iterable, Optional, Union

import numpy as np
import tensorflow as tf

from .backend import ArrayLike, Backend


class TensorflowBackend(Backend[tf.Tensor]):

    # basics

    def convert(self, value: ArrayLike, *, dtype: Optional[tf.DType] = None) -> tf.Tensor:
        if dtype is None:
            if isinstance(value, int):
                dtype = tf.int32
            if isinstance(value, float):
                dtype = tf.float32
            if isinstance(value, complex):
                dtype = tf.complex64
        return tf.convert_to_tensor(value, dtype=dtype)

    def compact(self, *values: ArrayLike) -> tuple[tf.Tensor, ...]:
        if len(values) == 0:
            return ()

        values = tuple(self.convert(value) for value in values)
        if len(values) == 1:
            return values

        dtype = get_compact_dtype(*(value.dtype for value in values))
        return tuple(tf.cast(value, dtype) for value in values)

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
        value0, value1 = self.compact(value0, value1)
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
        value0, value1 = self.compact(value0, value1)
        return value0 + value1

    def sub(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.compact(value0, value1)
        return value0 - value1

    def mul(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.compact(value0, value1)
        return value0 * value1

    def div(self, value0: ArrayLike, value1: ArrayLike) -> tf.Tensor:
        value0, value1 = self.compact(value0, value1)
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
        axis_pairs = np.asarray(tuple(zip(*axes)), dtype=int)  # [axes_n, 2]
        while len(axis_pairs) > 0:
            axis1, axis2 = axis_pairs[0]
            value = tf.experimental.numpy.trace(value, axis1=axis1, axis2=axis2)
            axis_pairs = axis_pairs[1:]
            axis_pairs = tf.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
            axis_pairs = tf.where(axis_pairs > axis2, axis_pairs - 1, axis_pairs)
        return value

    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> tf.Tensor:
        value = self.convert(value)
        axis_pairs = np.asarray(tuple(zip(*axes)), dtype=int)  # [axes_n, 2]
        while len(axis_pairs) > 0:
            axis1, axis2 = axis_pairs[0]
            value = tf.experimental.numpy.diagonal(value, axis1=axis2, axis2=axis2)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis2, axis_pairs - 1, axis_pairs)
        return value

    def dot(self,
        value0: ArrayLike, value1: ArrayLike, *,
        ndim0: int, ndim1: int,
        dot_axes: tuple[Iterable[int], Iterable[int]],
        bat_axes: tuple[Iterable[int], Iterable[int]],
    ) -> tuple[tf.Tensor, tuple[tuple[int, ...], tuple[int, ...]]]:
        value0, value1 = self.compact(value0, value1)

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

    # quantum operations

    def measure_pure_state(self,
        state: ArrayLike,
        batches_axes: Iterable[int],
        reduced_axes: Iterable[int],
        measure_axes: Iterable[int],
        measure_results: Optional[ArrayLike] = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        state = self.convert(state)
        state_shape = state.shape
        batches_axes = np.asarray(batches_axes, dtype=int)
        reduced_axes = np.asarray(reduced_axes, dtype=int)
        measure_axes = np.asarray(measure_axes, dtype=int)

        batches_shape = tf.gather(tf.shape(state), batches_axes)
        batches_n = tf.reduce_prod(batches_shape)
        choices_shape = tf.gather(tf.shape(state), measure_axes)
        choices_n = tf.reduce_prod(choices_shape)

        state = tf.transpose(state, [*batches_axes, *measure_axes, *reduced_axes])
        state = tf.reshape(state, [batches_n, choices_n, -1])
        # [batches_n, choices_n, reduced_n]

        probs = tf.math.abs(tf.math.conj(state) * state)
        probs = tf.reduce_sum(probs, axis=-1)
        # [batches_n, choices_n]

        if measure_results is not None:
            measure_results = tf.convert_to_tensor(measure_results, dtype=np.int32)
            # [(*batches_shape), choices_d], int32
            choice = tf_ravel_index(measure_results, choices_shape)
            # [(*batches_shape)], int32
            choice = tf.broadcast_to(choice, batches_shape)
            # [*batches_shape], int32
            choice = tf.reshape(choice, [-1])
            # [batches_n], int32
        else:
            choice = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
            choice = tf.squeeze(choice, axis=-1)
            # [batches_n], int32

        chosen_gather_indices = tf.stack([tf.range(batches_n, dtype=choice.dtype), choice], axis=-1)
        # [batches_n, 2], int32
        chosen_prob = tf.gather_nd(probs, chosen_gather_indices)
        # [batches_n]
        chosen_component = tf.gather_nd(state, chosen_gather_indices)
        # [batches_n, reduced_n]
        chosen_component /= tf.cast(
            tf.expand_dims(tf.sqrt(chosen_prob), axis=-1),
            dtype=chosen_component.dtype)  # normalization
        # [batches_n, reduced_n]

        chosen_onehot = tf.one_hot(choice, choices_n, dtype=state.dtype)
        # [batches_n, choices_n]
        chosen_component = tf.expand_dims(chosen_component, axis=-2)
        chosen_onehot = tf.expand_dims(chosen_onehot, axis=-1)
        chosen_state = chosen_component * chosen_onehot
        # [batches_n, choices_n, reduced_n]

        choice = tf.unravel_index(choice, choices_shape)
        # [batches_n, choices_d]
        choice = tf.reshape(choice, [*batches_shape, len(measure_axes)])
        # [*batches_shape, choices_d]
        chosen_prob = tf.reshape(chosen_prob, batches_shape)
        # [*batches_shape]
        chosen_state = tf.reshape(chosen_state, state_shape)
        # [*batches_shape, *measure_shape, *reduced_shape]

        return choice, chosen_prob, chosen_state

    def measure_mixed_state(self,
        state: ArrayLike,
        batches_axes: Iterable[int],
        reduced_axes: Iterable[tuple[int, int]],
        measure_axes: Iterable[tuple[int, int]],
        measure_results: Optional[ArrayLike] = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        state = self.convert(state)
        state_shape = state.shape
        batches_axes = np.asarray(batches_axes, dtype=int)
        reduced_axes = np.asarray(reduced_axes, dtype=int)
        measure_axes = np.asarray(measure_axes, dtype=int)

        batches_shape = tf.gather(tf.shape(state), batches_axes)
        batches_n = tf.reduce_prod(batches_shape)
        reduced_shape = tf.gather(tf.shape(state), reduced_axes[:, 0])
        reduced_n = tf.reduce_prod(reduced_shape)
        choices_shape = tf.gather(tf.shape(state), measure_axes[:, 0])
        choices_n = tf.reduce_prod(choices_shape)

        state = tf.transpose(state, [
            *batches_axes, *measure_axes[:, 0], *measure_axes[:, 1], *reduced_axes[:, 0], *reduced_axes[:, 1]])
        state = tf.reshape(state, [batches_n, choices_n, choices_n, reduced_n, reduced_n])
        # [batches_n, choices_n, choices_n, reduced_n, reduced_n]

        probs = tf.linalg.trace(state)  # axis1=-1, axis2=-2
        # [batches_n, choices_n, choices_n]
        probs = tf.linalg.diag_part(probs)  # axis1=-2, axis2=-1
        # [batches_n, choices_n]

        if measure_results is not None:
            measure_results = tf.convert_to_tensor(measure_results, dtype=np.int32)
            # [(*batches_shape), choices_d], int32
            choice = tf_ravel_index(measure_results, choices_shape)
            # [(*batches_shape)], int32
            choice = tf.broadcast_to(choice, batches_shape)
            # [*batches_shape], int32
            choice = tf.reshape(choice, [-1])
            # [batches_n], int32
        else:
            choice = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
            choice = tf.squeeze(choice, axis=-1)
            # [batches_n], int32

        chosen_prob_gather_indices = tf.stack([tf.range(batches_n, dtype=choice.dtype), choice], axis=-1)
        # [batches_n, 2], int32
        chosen_prob = tf.gather_nd(probs, chosen_prob_gather_indices)
        # [batches_n]

        chosen_component_gather_indices = tf.stack([tf.range(batches_n, dtype=choice.dtype), choice, choice], axis=-1)
        # [batches_n, 3], int32
        chosen_component = tf.gather_nd(state, chosen_component_gather_indices)
        # [batches_n, reduced_n, reduced_n]
        chosen_component /= tf.cast(
            tf.expand_dims(tf.expand_dims(chosen_prob, axis=-1), axis=-1),
            dtype=chosen_component.dtype)  # normalization
        # [batches_n, reduced_n, reduced_n]

        chosen_onehot_indices = choice + choice * choices_n
        chosen_onehot = tf.one_hot(chosen_onehot_indices, choices_n * choices_n, dtype=state.dtype)
        chosen_onehot = tf.reshape(chosen_onehot, [batches_n, choices_n, choices_n])
        # [batches_n, choices_n, choices_n]

        chosen_component = tf.expand_dims(tf.expand_dims(chosen_component, axis=-3), axis=-3)
        chosen_onehot = tf.expand_dims(tf.expand_dims(chosen_onehot, axis=-1), axis=-1)
        chosen_state = chosen_component * chosen_onehot
        # [batches_n, choices_n, choices_n, reduced_n, reduced_n]

        choice = tf.unravel_index(choice, choices_shape)
        # [batches_n, choices_d]
        choice = tf.reshape(choice, [*batches_shape, len(measure_axes)])
        # [*batches_shape, choices_d]
        chosen_prob = tf.reshape(chosen_prob, batches_shape)
        # [*batches_shape]
        chosen_state = tf.reshape(chosen_state, state_shape)
        # [*batches_shape, *measure_shape, *measure_shape, *reduced_shape, *reduced_shape]

        return choice, chosen_prob, chosen_state


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


def tf_ravel_index(index: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
    # index: [...,shape_d]
    # shape: [shape_d]
    # result: [...]
    shape_cumprod = tf.math.cumprod(shape, reverse=True)
    shape_cumprod = tf.concat([shape_cumprod[1:], tf.constant([1], dtype=shape_cumprod.dtype)], axis=0)
    return tf.reduce_sum(index * shape_cumprod, axis=-1)
