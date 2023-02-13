from contextvars import ContextVar

from .backend import Backend
from .numpy_backend import numpy_backend

_backend_stack_context_var: ContextVar[tuple[Backend, ...]] = ContextVar("backend_stack", default=(numpy_backend,))


def get_default_backend() -> Backend:
    return _backend_stack_context_var.get()[-1]


def set_default_backend(backend: Backend):
    backend_stack = _backend_stack_context_var.get()
    _backend_stack_context_var.set((*backend_stack[:-1], backend))


def push_context_backend(backend: Backend):
    backend_stack = _backend_stack_context_var.get()
    _backend_stack_context_var.set((*backend_stack, backend))


def pop_context_backend() -> Backend:
    backend_stack = _backend_stack_context_var.get()
    if not len(backend_stack) > 1:
        raise RuntimeError("No enough backends in the context stack to pop!")
    _backend_stack_context_var.set(backend_stack[:-1])
    return backend_stack[-1]
