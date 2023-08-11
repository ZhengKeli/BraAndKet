import math
from typing import Any, Callable, Iterable, Union

from braandket.backend import Backend, BackendValue, get_default_backend
from braandket.space import HSpace, KetSpace, NumSpace, Space
from .special import NumericTensor, OperatorTensor, PureStateTensor
from .tensor import QTensor

# constants

e = math.e
pi = math.pi


# constructors

def zero() -> NumericTensor:
    return NumericTensor.of(0, ())


def one() -> NumericTensor:
    return NumericTensor.of(1, ())


def zeros(space: NumSpace) -> NumericTensor:
    backend = get_default_backend()
    values = backend.zeros((space.n,))
    return NumericTensor(values, (space,))


def ones(space: NumSpace) -> NumericTensor:
    backend = get_default_backend()
    values = backend.ones((space.n,))
    return NumericTensor(values, (space,))


def eigenstate(space: KetSpace, index: int) -> PureStateTensor:
    backend = get_default_backend()
    values = backend.onehot(index, space.n)
    return PureStateTensor(values, (space,))


def operator(space: KetSpace, ket_index: int, bra_index: int) -> OperatorTensor:
    ket_vector = eigenstate(space, ket_index)
    bra_vector = eigenstate(space, bra_index).ct
    return OperatorTensor.of(ket_vector @ bra_vector)


def projector(space: KetSpace, index: int) -> OperatorTensor:
    return operator(space, index, index)


def identity(space: KetSpace) -> OperatorTensor:
    backend = get_default_backend()
    values = backend.eye(space.n)
    return OperatorTensor(values, (space, space.ct))


# prod & sum

def prod(*items: QTensor) -> QTensor:
    if len(items) == 0:
        return one()
    x = items[0]
    for item in items[1:]:
        x = x @ item
    return x


def sum(*items: QTensor) -> QTensor:
    if len(items) == 0:
        return zero()
    x = items[0]
    for item in items[1:]:
        x = x + item
    return x


def sum_ct(*items: QTensor) -> QTensor:
    s = sum(*items)
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
    with tensor.backend:
        return prod(tensor, *(space.identity() for space in spaces))


# numeric

def _construct_wrapped_unary_op(func: Callable[[BackendValue], BackendValue]):
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
