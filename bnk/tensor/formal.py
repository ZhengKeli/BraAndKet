import abc
from typing import Tuple, Union, Iterable, Set

import numpy as np

from .abstract import QTensor
from ..space import Space, NumSpace, HSpace, KetSpace, BraSpace


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
            scalar = self.scalar()
            if scalar == 0:
                return other
            return scalar + other
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
            scalar = self.scalar()
            if scalar == 1:
                return other
            return scalar * other
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

    def broadcast(self, spaces: Iterable[Space]):
        ket_spaces = []
        num_spaces = []

        spaces = set(spaces)
        while spaces:
            space = spaces.pop()

            if space in self.spaces:
                continue

            if isinstance(space, NumSpace):
                num_spaces.append(space)
                continue

            if isinstance(space, HSpace):
                if space.ct not in spaces:
                    raise TypeError(f"Can not broadcast unpaired space {space}.")
                spaces.remove(space.ct)
                ket_spaces.append(space.ket)
                continue

            raise TypeError(f"Unsupported custom type {type(space)}!")

        return self._formal_broadcast(ket_spaces, num_spaces)

    @abc.abstractmethod
    def _formal_flatten(self, ket_spaces, bra_spaces):
        pass

    def flatten(self, ket_spaces=None, bra_spaces=None, *, return_spaces=False):
        _ket_spaces = [space for space in self.spaces if isinstance(space, KetSpace)]
        _ket_spaces = tuple(sorted(_ket_spaces, key=lambda sp: (-sp.n, id(sp))))

        _bra_spaces = [space for space in self.spaces if isinstance(space, BraSpace)]
        _bra_spaces = tuple(sorted(_bra_spaces, key=lambda sp: (-sp.n, id(sp.ket))))

        if ket_spaces is None:
            ket_spaces = _ket_spaces
        else:
            ket_spaces = tuple(ket_spaces)
            ket_spaces_set = set(ket_spaces)
            remained_ket_spaces = (space for space in _ket_spaces if space not in ket_spaces_set)
            ket_spaces = (*ket_spaces, *remained_ket_spaces)

        if bra_spaces is None:
            bra_spaces = _bra_spaces
        else:
            bra_spaces = tuple(bra_spaces)
            bra_spaces_set = set(bra_spaces)
            remained_ket_spaces = (space for space in _ket_spaces if space not in bra_spaces_set)
            ket_spaces = (*ket_spaces, *remained_ket_spaces)

        flattened = self._formal_flatten(ket_spaces, bra_spaces)

        if return_spaces:
            return (ket_spaces, bra_spaces), flattened
        else:
            return flattened

    @classmethod
    @abc.abstractmethod
    def _formal_inflate(cls, flattened, ket_spaces, bra_spaces, *, copy=True):
        pass

    @classmethod
    def inflate(cls, flattened, ket_spaces: Iterable[KetSpace], bra_spaces: Iterable[BraSpace], *, copy=True):
        ket_spaces = tuple(ket_spaces)
        bra_spaces = tuple(bra_spaces)
        return cls._formal_inflate(flattened, ket_spaces, bra_spaces, copy=copy)
