from tqdm import tqdm

from ..tensor import QTensor


class Evolution:
    def __init__(self,
            kernel, *,
            check_dt=None, check_func=None,
            rectify=True, verbose=True,
    ):
        self._kernel = kernel
        self._check_dt = check_dt
        self._check_func = check_func
        self._rectify = rectify
        self._verbose = verbose

    @property
    def time(self):
        return self._kernel.time

    @time.setter
    def time(self, time):
        self._kernel.time = time

    @property
    def value(self):
        return self._kernel.value

    @value.setter
    def value(self, value):
        self._kernel.value = value

    def evolve(self, span, **kwargs):
        check_dt = self._check_dt
        if check_dt is None:
            check_dt = span

        verbose = self._verbose
        if verbose:
            verbose = tqdm(total=span)

        self.check()
        remain = span
        while remain > check_dt:
            remain -= check_dt
            self._kernel.evolve(check_dt, **kwargs)
            self.check()
            if verbose:
                verbose.update(check_dt)

        if remain > 0:
            self._kernel.evolve(remain, **kwargs)
            self.check()
            if verbose:
                verbose.update(remain)

    def check(self):
        self._kernel.rectify()

        check_func = self._check_func
        if not callable(check_func):
            return

        correct = check_func(self.time, self.value)
        if correct is None:
            return

        if not isinstance(correct, QTensor):
            raise TypeError(
                "the return value of check_func is expected to be a QTensor, "
                f"got {correct}")

        self.value = correct
        self._kernel.rectify()
