import abc

import numpy as np


# kernel

class Kernel(abc.ABC):
    def __init__(self, model, time, value, **kwargs):
        model, value, wrapping = self.init_model(model, value)

        self._wrapping = wrapping
        self._model = model
        self._time = time
        self._value = value
        self._args = kwargs

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def value(self):
        return self.wrap_value(self._value, self._wrapping)

    @value.setter
    def value(self, value):
        self._value = self.unwrap_value(value, self._wrapping)

    @classmethod
    @abc.abstractmethod
    def init_model(cls, model, value):
        pass

    @classmethod
    @abc.abstractmethod
    def unwrap_value(cls, value, wrapping):
        pass

    @classmethod
    @abc.abstractmethod
    def wrap_value(cls, value, wrapping):
        pass

    def evolve(self, span, **kwargs):
        model = self._model
        t = self._time
        value = self._value

        args = {**self._args}
        for k, arg in kwargs.items():
            if args.get(k) is None:
                args[k] = arg

        value = self.compute(model, t, value, span, **args)
        t = t + span

        self._time = t
        self._value = value

    @classmethod
    @abc.abstractmethod
    def compute(cls, model, t, value, span, **kwargs):
        pass

    def rectify(self):
        self._value = self.normalize(self._value)

    @classmethod
    @abc.abstractmethod
    def normalize(cls, value):
        pass


# static

class StaticMixin(Kernel, abc.ABC):

    @classmethod
    def init_model(cls, model, value):
        (hb, hmt, deco, dynamic_hmt, dynamic_deco), value, wrapping = super().init_model(model, value)
        if len(dynamic_hmt) > 0:
            raise ValueError("Static kernel can NOT deal with dynamic hamiltonian.")
        if len(dynamic_deco) > 0:
            raise ValueError("Static kernel can NOT deal with dynamic decoherence.")
        return (hb, hmt, deco), value, wrapping

    @classmethod
    def compute(cls, model, _, value, span, **kwargs):
        return cls.compute_static(model, value, span, **kwargs)

    @classmethod
    @abc.abstractmethod
    def compute_static(cls, model, value, span, **kwargs):
        pass


class StaticSteppingMixin(StaticMixin):
    def __init__(self, model, time, value, *, dt=None, **kwargs):
        super().__init__(model, time, value, dt=dt, **kwargs)

    @classmethod
    def compute_static(cls, model, value, span, *, dt=None, **kwargs):
        if dt is None:
            raise ValueError("Parameter dt is required")

        tiled_n = int(np.ceil(span / dt))
        tiled_dt = span / tiled_n
        return cls.compute_static_stepping(model, value, tiled_n, tiled_dt)

    @classmethod
    @abc.abstractmethod
    def compute_static_stepping(cls, model, value, n, dt, **kwargs):
        pass
