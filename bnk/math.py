from .tensor import QTensor, one, zero
from .utils import structured_iter


def prod(*items):
    x = one
    for item in structured_iter(items):
        if isinstance(item, QTensor):
            x = x @ item
        else:
            x = x * item
    return x


def sum(*items):
    x = zero
    for item in structured_iter(items):
        x += item
    return x


def sum_ct(*items):
    s = sum(*items)
    return s + s.ct
