from typing import Iterable, Optional, Union

import numpy as np

from .backend import ArrayLike, Backend


class NumpyBackend(Backend[np.ndarray]):

    # basics

    def convert(self, value: ArrayLike, *, dtype=None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    def compact(self, *values: ArrayLike) -> tuple[np.ndarray, ...]:
        return tuple(self.convert(value) for value in values)

    def copy(self, value: ArrayLike) -> np.ndarray:
        return np.copy(value)

    # constructors

    def zeros(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.zeros(tuple(shape), dtype=dtype)

    def ones(self, shape: Iterable[int], *, dtype=np.float32) -> np.ndarray:
        return np.ones(tuple(shape), dtype=dtype)

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
        return np.reshape(value, tuple(shape))

    def transpose(self, value: ArrayLike, *, axes: Iterable[int]) -> np.ndarray:
        return np.transpose(value, tuple(axes))

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
        axis_pairs = np.asarray(tuple(zip(*axes)), dtype=int)  # [axes_n, 2]
        while len(axis_pairs) > 0:
            axis1, axis2 = axis_pairs[0]
            value = np.trace(value, axis1=axis1, axis2=axis2)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis2, axis_pairs - 1, axis_pairs)
        return value

    def diag(self, value: ArrayLike, axes: tuple[Iterable[int], Iterable[int]]) -> np.ndarray:
        axis_pairs = np.asarray(tuple(zip(*axes)), dtype=int)  # [axes_n, 2]
        while len(axis_pairs) > 0:
            axis1, axis2 = axis_pairs[0]
            value = np.diagonal(value, axis1=axis1, axis2=axis2)
            axis_pairs = axis_pairs[1:]
            axis_pairs = np.where(axis_pairs > axis1, axis_pairs - 1, axis_pairs)
            axis_pairs = np.where(axis_pairs > axis2, axis_pairs - 1, axis_pairs)
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

    # quantum operations

    def measure_pure_state(self,
        state: ArrayLike,
        batches_axes: Iterable[int],
        reduced_axes: Iterable[int],
        measure_axes: Iterable[int],
        measure_results: Optional[Iterable[ArrayLike]] = None,
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        state = self.convert(state)
        state_shape = state.shape
        batches_axes = np.asarray(batches_axes, dtype=int)
        reduced_axes = np.asarray(reduced_axes, dtype=int)
        measure_axes = np.asarray(measure_axes, dtype=int)

        choices_shape = np.shape(state)[measure_axes]
        choices_n = np.prod(choices_shape)
        batches_shape = np.shape(state)[batches_axes]
        batches_n = np.prod(batches_shape)

        state = np.transpose(state, [*batches_axes, *measure_axes, *reduced_axes])
        state = np.reshape(state, [batches_n, choices_n, -1])
        # [batches_n, choices_n, reduced_n]

        probs = np.abs(np.conj(state) * state)
        probs = np.sum(probs, axis=-1)
        # [batches_n, choices_n]

        if measure_results is not None:
            measure_results = np.asarray(measure_results, dtype=np.int32)
            choice = np.ravel_multi_index(measure_results, choices_shape)
            choice = np.broadcast_to(choice, batches_shape)
            choice = np.reshape(choice, [-1])
            # [batches_n], int32
        else:
            choice = np.apply_along_axis(lambda ps: np.random.choice(choices_n, p=ps), 0, probs)
            choice = np.asarray(choice, dtype=np.int32)
            # [batches_n], int32

        chosen_prob = probs[np.arange(batches_n), choice]
        # [batches_n]
        chosen_component = state[np.arange(batches_n), choice]
        # [batches_n, reduced_n]
        chosen_component /= np.expand_dims(chosen_prob, axis=-1)  # normalization
        # [batches_n, reduced_n]

        chosen_onehot = np.arange(choices_n, dtype=choice.dtype) == np.expand_dims(choice, -1)
        chosen_onehot = np.asarray(chosen_onehot, dtype=state.dtype)
        # [batches_n, choices_n]
        chosen_component = np.expand_dims(chosen_component, axis=-2)
        chosen_onehot = np.expand_dims(chosen_onehot, axis=-1)
        chosen_state = chosen_component * chosen_onehot
        # [batches_n, choices_n, reduced_n]

        chosen_state = np.reshape(chosen_state, state_shape)
        # [*batches_shape, *measure_shape, *reduced_shape]
        choice = np.unravel_index(choice, choices_shape)
        # [choices_d]

        return choice, chosen_prob, chosen_state

    def measure_mixed_state(self,
        state: ArrayLike,
        batches_axes: Iterable[int],
        reduced_axes: Iterable[tuple[int, int]],
        measure_axes: Iterable[tuple[int, int]],
        measure_results: Optional[Iterable[ArrayLike]] = None,
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        state = self.convert(state)
        state_shape = state.shape
        batches_axes = np.asarray(batches_axes, dtype=int)
        reduced_axes = np.asarray(reduced_axes, dtype=int)
        measure_axes = np.asarray(measure_axes, dtype=int)

        choices_shape = np.shape(state)[measure_axes[:, 0]]
        choices_n = np.prod(choices_shape)
        reduced_shape = np.shape(state)[reduced_axes[:, 0]]
        reduced_n = np.prod(reduced_shape)
        batches_shape = np.shape(state)[batches_axes]
        batches_n = np.prod(batches_shape)

        state = np.transpose(state, [
            *batches_axes, *measure_axes[:, 0], *measure_axes[:, 1], *reduced_axes[:, 0], *reduced_axes[:, 1]])
        state = np.reshape(state, [batches_n, choices_n, choices_n, reduced_n, reduced_n])
        # [batches_n, choices_n, choices_n, reduced_n, reduced_n]

        probs = np.trace(state, axis1=-1, axis2=-2)
        # [batches_n, choices_n, choices_n]
        probs = np.diagonal(probs, axis1=-2, axis2=-1)
        # [batches_n, choices_n]

        if measure_results is not None:
            measure_results = np.asarray(measure_results, dtype=np.int32)
            choice = np.ravel_multi_index(measure_results, choices_shape)
            choice = np.broadcast_to(choice, batches_shape)
            choice = np.reshape(choice, [-1])
            # [batches_n], int32
        else:
            choice = np.apply_along_axis(lambda ps: np.random.choice(choices_n, p=ps), 0, probs)
            choice = np.asarray(choice, dtype=np.int32)
            # [batches_n], int32

        chosen_prob = probs[np.arange(batches_n), choice]
        # [batches_n]
        chosen_component = state[np.arange(batches_n), choice, choice]
        # [batches_n, reduced_n, reduced_n]
        chosen_component /= np.expand_dims(chosen_prob, axis=[-2, -1])  # normalization
        # [batches_n, reduced_n, reduced_n]

        chosen_onehot = np.zeros([batches_n, choices_n, choices_n], dtype=state.dtype)
        chosen_onehot[:, choice, choice] = 1.0
        # [batches_n, choices_n, choices_n]

        chosen_component = np.expand_dims(chosen_component, axis=[-3, -4])
        chosen_onehot = np.expand_dims(chosen_onehot, axis=[-1, -2])
        chosen_state = chosen_component * chosen_onehot
        # [batches_n, choices_n, choices_n, reduced_n, reduced_n]

        chosen_state = np.reshape(chosen_state, state_shape)
        # [*batches_shape, *measure_shape, *measure_shape, *reduced_shape, *reduced_shape]
        choice = np.unravel_index(choice, choices_shape)
        # [batches_n, choices_d]

        return choice, chosen_prob, chosen_state


numpy_backend = NumpyBackend()
