import abc
from typing import Optional


class Space(abc.ABC):

    # basics

    @property
    @abc.abstractmethod
    def n(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        pass

    # str & repr

    def __repr__(self):
        if self.name is None:
            return f"<{type(self).__name__}: n={self.n}>"
        else:
            return f"<{type(self).__name__}: n={self.n}, name={self.name}>"

    def __str__(self):
        return repr(self)
