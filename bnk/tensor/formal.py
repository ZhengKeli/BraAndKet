import abc
from typing import Tuple, Union, Iterable, Set

import numpy as np

from .abstract import QTensor
from ..space import KetSpace
from ..space import Space


class FormalQTensor(QTensor, abc.ABC):

    # basic

    def __init__(self, spaces: Iterable[Space]):
        spaces_set = set()
        for space in spaces:
            if space in spaces_set:
                raise ValueError("There are duplicated spaces!")
            spaces_set.add(space)
        self._formal_spaces = spaces_set

    @property
    def spaces(self) -> Set[Space]:
        return self._formal_spaces

    @abc.abstractmethod
    def _formal_getitem(self, *items: Tuple[Space, Union[int, slice, tuple]]):
        """

        :param items: tuples of space and corresponding slice
        :return: values. The type is recommended to be numpy.ndarray or any compatible data types.
        """

    def __getitem__(self, items):
        if isinstance(items, dict):
            items = items.items()

        if isinstance(items, Space):
            formal_items = ((items, slice(None)),)
            return self._formal_getitem(*formal_items)

        items = tuple(items)

        if len(items) == 0:
            return self._formal_getitem()

        formal_items = []
        for item in items:
            if isinstance(item, Space):
                item = (item, slice(None))
            else:
                item = tuple(item)
                if len(item) != 2:
                    formal_items = None
                    break
                spa, sli = item
                if not isinstance(spa, Space):
                    formal_items = None
                    break
                if not isinstance(sli, (int, slice, tuple)):
                    try:
                        sli = int(sli)
                    except TypeError:
                        try:
                            sli = tuple(int(i) for i in sli)
                        except TypeError:
                            raise TypeError(f"Not supported indexing with {sli}")
                item = spa, sli
            formal_items.append(item)

        if formal_items is None:
            item = items
            if len(item) != 2:
                raise ValueError("Unsupported argument for getting item: " + str(item))
            spa, sli = item
            if not isinstance(spa, Space):
                raise ValueError("Unsupported argument for getting item: " + str(item))
            formal_items = (item,)

        return self._formal_getitem(*formal_items)

    # linear operations

    @abc.abstractmethod
    def _formal_add(self, other):
        """

        :param other: other tensor after broadcast
        :return: result of adding
        """
        pass

    def __add__(self, other):
        if other == 0:
            return self
        if self.is_scalar:
            from .numpy import NumpyQTensor
            return NumpyQTensor.from_scalar(self.scalar() + other)
        if not isinstance(other, QTensor):
            raise TypeError(f"Can not perform + operation with {other}: this QTensor is not zero-dimensional.")

        if self.spaces == other.spaces:
            return self._formal_add(other)

        spaces_broadcast = self.spaces.union(other.spaces)
        self_broadcast = self.broadcast(spaces_broadcast)
        other_broadcast = other.broadcast(spaces_broadcast)
        return self_broadcast + other_broadcast

    @abc.abstractmethod
    def _formal_mul(self, other):
        """

        :param new_spaces: new spaces
        :param self: self tensor
        :param other: multiplying scalar
        :return: result of multiplication
        """

    def __mul__(self, other):
        if other == 1:
            return self
        if self.is_scalar:
            from .numpy import NumpyQTensor
            return NumpyQTensor.from_scalar(self.scalar() * other)
        if isinstance(other, QTensor):
            if not other.is_scalar:
                raise TypeError("Please use matmul operator \"@\" for tensor product.")
            other = other.scalar()
        else:
            other = np.asarray(other).item()

        return self._formal_mul(other)

    # tensor operations

    @abc.abstractmethod
    def _formal_trace(self, *spaces: KetSpace):
        """

        :param spaces: the spaces to be traced (not empty)
        :return: trace result
        """

    def trace(self, *spaces: KetSpace):
        if len(spaces) == 0:
            spaces = self.spaces
        return self._formal_trace(*spaces)

    @abc.abstractmethod
    def _formal_matmul(self, other):
        """

        :param other: an other tensor
        :return: tensor product
        """

    def __matmul__(self, other):
        if not isinstance(other, QTensor):
            return self * other
        return self._formal_matmul(other)

    # space operations

    @abc.abstractmethod
    def _formal_broadcast(self, ket_spaces: Iterable[KetSpace], num_spaces: Iterable[NumSpace]):
        pass
