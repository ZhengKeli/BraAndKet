import unittest

import braandket as bnk


class TestSpace(unittest.TestCase):

    def test00_num_space(self):
        num_space = bnk.NumSpace(7)
        print(num_space)

        num_space_named = bnk.NumSpace(8, name='batch')
        print(num_space_named)

    def test01_ket_space(self):
        ket_space = bnk.KetSpace(3)
        print(ket_space)

        ket_space_named = bnk.KetSpace(3, name='a')
        print(ket_space_named)

    def test02_bra_space(self):
        ket_space = bnk.KetSpace(3)
        bra_space = ket_space.ct
        print(bra_space)

        ket_space_named = bnk.KetSpace(3, name='a')
        bra_space_named = ket_space_named.ct
        print(bra_space_named)

    def test03_bra_space_singleton(self):
        ket_space = bnk.KetSpace(3, name='k')
        bra_space = ket_space.ct
        self.assertIs(bra_space.ct, ket_space, "bra_space.ct != ket_space")

        bra_space_constructed = bnk.BraSpace(ket_space)
        self.assertIs(bra_space, bra_space_constructed, "BraSpace(ket_space) != ket_space.ct")
