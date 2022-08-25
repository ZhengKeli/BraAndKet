import unittest

from braandket.space import BraSpace, KetSpace, NumSpace


class TestSpace(unittest.TestCase):

    def test00_num_space(self):
        num_space = NumSpace(7)
        print(num_space)

        num_space_named = NumSpace(8, name='batch')
        print(num_space_named)

    def test01_ket_space(self):
        ket_space = KetSpace(3)
        print(ket_space)

        ket_space_named = KetSpace(3, name='a')
        print(ket_space_named)

    def test02_bra_space(self):
        ket_space = KetSpace(3)
        bra_space = ket_space.ct
        print(bra_space)

        ket_space_named = KetSpace(3, name='a')
        bra_space_named = ket_space_named.ct
        print(bra_space_named)

    def test03_bra_space_singleton(self):
        ket_space = KetSpace(3, name='k')
        bra_space = ket_space.ct
        self.assertIs(bra_space.ct, ket_space, "bra_space.ct != ket_space")

        bra_space_constructed = BraSpace(ket_space)
        self.assertIs(bra_space, bra_space_constructed, "BraSpace(ket_space) != ket_space.ct")
