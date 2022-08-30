from .backend import Backend, ValuesType
from .numpy_backend import NumpyBackend, numpy_backend


_default_backend = numpy_backend


def get_default_backend() -> Backend:
    return _default_backend


def set_default_backend(backend: Backend):
    global _default_backend
    _default_backend = backend
