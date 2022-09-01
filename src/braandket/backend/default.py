from .backend import Backend
from .numpy_backend import numpy_backend


# default

_default_backend = numpy_backend


def get_default_backend() -> Backend:
    return _default_backend


def set_default_backend(backend: Backend):
    global _default_backend
    _default_backend = backend


# context

_context_backend_stack = []


def push_context_backend(backend: Backend):
    global _context_backend_stack
    _context_backend_stack.append(get_default_backend())
    set_default_backend(backend)


def pop_context_backend() -> Backend:
    global _context_backend_stack
    backend = _context_backend_stack.pop()
    set_default_backend(backend)
    return backend

# TODO thread-safe?
