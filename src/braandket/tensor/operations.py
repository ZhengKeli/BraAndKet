import math
from typing import Any, Callable, Iterable, Optional, Union

from braandket.backend import Backend, ValuesType, get_default_backend
from braandket.space import HSpace, KetSpace, NumSpace, Space
from .special import NumericTensor, OperatorTensor, PureStateTensor
from .tensor import QTensor

# constants

e = math.e
pi = math.pi


# constructors

def zero(*, backend: Optional[Backend] = None) -> NumericTensor:
    return NumericTensor.of(0, (), backend=backend)


def one(*, backend: Optional[Backend] = None) -> NumericTensor:
    return NumericTensor.of(1, (), backend=backend)


def zeros(space: NumSpace, *, backend: Optional[Backend] = None) -> NumericTensor:
    backend = backend or get_default_backend()
    values = backend.zeros((space.n,))
    return NumericTensor(values, (space,), backend)


def ones(space: NumSpace, *, backend: Optional[Backend] = None) -> NumericTensor:
    backend = backend or get_default_backend()
    values = backend.ones((space.n,))
    return NumericTensor(values, (space,), backend)


def eigenstate(space: KetSpace, index: int, *, backend: Optional[Backend] = None) -> PureStateTensor:
    backend = backend or get_default_backend()
    values = backend.onehot(index, space.n)
    return PureStateTensor(values, (space,), backend)


def operator(space: KetSpace, ket_index: int, bra_index: int, *, backend: Optional[Backend] = None) -> OperatorTensor:
    ket_vector = eigenstate(space, ket_index, backend=backend)
    bra_vector = eigenstate(space, bra_index, backend=backend).ct
    return OperatorTensor.of(ket_vector @ bra_vector)


def projector(space: KetSpace, index: int, *, backend: Optional[Backend] = None) -> OperatorTensor:
    return operator(space, index, index, backend=backend)


def identity(space: KetSpace, *, backend: Optional[Backend] = None) -> OperatorTensor:
    backend = backend or get_default_backend()
    values = backend.eye(space.n)
    return OperatorTensor(values, (space, space.ct), backend)


# prod & sum

def prod(*items: QTensor, backend: Optional[Backend] = None) -> QTensor:
    if len(items) == 0:
        return one(backend=backend)
    x = items[0]
    for item in items[1:]:
        x = x @ item
    return x


def sum(*items: QTensor, backend: Optional[Backend] = None) -> QTensor:
    if len(items) == 0:
        return zero(backend=backend)
    x = items[0]
    for item in items[1:]:
        x = x + item
    return x


def sum_ct(*items, backend: Optional[Backend] = None) -> QTensor:
    s = sum(*items, backend=backend)
    return s + s.ct


# broadcast

def _broadcast(tensor: QTensor, *spaces: Space) -> QTensor:
    # extract non-existed spaces
    spaces = tuple(space for space in spaces if space not in tensor.spaces)

    # shortcut if there is no non-existed spaces
    if len(spaces) == 0:
        return tensor

    # construct new tensor
    new_spaces = (*spaces, *tensor._spaces)
    new_values = tensor.backend.expand(tensor._values, axes=range(len(spaces)), sizes=(space.n for space in spaces))
    return tensor.spawn(new_values, new_spaces)


def _broadcast_num_spaces(tensor0: QTensor, tensor1: QTensor, *tensors: QTensor) -> tuple[QTensor, ...]:
    if len(tensors) == 0:
        # extract num_spaces
        tensor0_num_spaces = tuple(space for space in tensor0.spaces if isinstance(space, NumSpace))
        tensor1_num_spaces = tuple(space for space in tensor1.spaces if isinstance(space, NumSpace))

        # broadcast num_spaces
        tensor0 = _broadcast(tensor0, *tensor1_num_spaces)
        tensor1 = _broadcast(tensor1, *tensor0_num_spaces)

        return tensor0, tensor1
    else:
        tensors = [tensor0, tensor1, *tensors]
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                tensors[i], tensors[j] = _broadcast_num_spaces(tensors[i], tensors[j])
        return tuple(tensors)


def _broadcast_h_spaces(tensor0: QTensor, tensor1: QTensor) -> tuple[QTensor, QTensor]:
    # extract h_spaces
    tensor0_h_spaces = tuple(space for space in tensor0.spaces if isinstance(space, HSpace))
    tensor1_h_spaces = tuple(space for space in tensor1.spaces if isinstance(space, HSpace))

    # broadcast h_spaces
    if tensor0_h_spaces != tensor1_h_spaces:
        if len(tensor0_h_spaces) == 0:
            tensor0 = _broadcast(tensor0, *tensor1_h_spaces)
        elif len(tensor1_h_spaces) == 0:
            tensor1 = _broadcast(tensor1, *tensor0_h_spaces)
        else:
            raise ValueError("Can not broadcast tensors with different set of h_spaces set.")

    return tensor0, tensor1


def _expand_with_identities(tensor: OperatorTensor, *spaces: KetSpace):
    return prod(tensor, *(space.identity(backend=tensor.backend) for space in spaces))


# numeric

def _construct_wrapped_unary_op(func: Callable[[ValuesType], ValuesType]):
    def wrapped_op(value: Union[NumericTensor, Any]) -> NumericTensor:
        value = NumericTensor.of(value)
        op = getattr(value.backend, func.__name__)
        # noinspection PyProtectedMember
        return value.spawn(op(value._values), value._spaces)

    return wrapped_op


pow = _construct_wrapped_unary_op(Backend.pow)
square = _construct_wrapped_unary_op(Backend.square)
sqrt = _construct_wrapped_unary_op(Backend.sqrt)
exp = _construct_wrapped_unary_op(Backend.exp)
sin = _construct_wrapped_unary_op(Backend.sin)
cos = _construct_wrapped_unary_op(Backend.cos)
abs = _construct_wrapped_unary_op(Backend.abs)


# choose

def choose(probs: Iterable[Union[NumericTensor, Any]]) -> NumericTensor:
    probs = tuple(NumericTensor.of(prob) for prob in probs)
    if len(probs) == 0:
        raise ValueError("Can not choose from empty set.")

    probs = _broadcast_num_spaces(*probs)
    spaces = tuple(probs[0].spaces)
    backend = probs[0].backend

    probs_values = tuple(prob.values(*spaces) for prob in probs)
    choice_values = backend.choose(probs_values)
    choice = NumericTensor.of(choice_values, spaces, backend=backend)
    return choice


def take(values: Iterable[QTensor], indices: NumericTensor) -> QTensor:
    values = _broadcast_num_spaces(*values)
    # TODO automatically expand for OperatorTensor
    assert len(values) > 0
    spaces = tuple(values[0].spaces)
    backend = values[0].backend
    for value in values:
        indices, _ = _broadcast_h_spaces(indices, value)

    values_values = tuple(value.values(*spaces) for value in values)
    indices_values = indices.values(*spaces)
    taken_values = backend.take(values_values, indices_values)
    taken = NumericTensor.of(taken_values, spaces, backend=backend)
    return taken
