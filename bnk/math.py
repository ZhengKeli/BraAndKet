from .tensor import QTensor, one, zero


def prod(*items):
    x = one
    for item in items:
        if isinstance(item, QTensor):
            x = x @ item
        else:
            x = x * item
    return x


def sum(*items):
    x = zero
    for item in items:
        x += item
    return x
