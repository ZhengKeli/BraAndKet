import unittest

from braandket.model.model import QParticle


class TestModel(unittest.TestCase):
    def test00_model(self):
        particle1 = QParticle(2, name="p1")
        particle2 = QParticle(2, name="p2")
        print(particle1.state)
        print(particle2.state)

        system = particle1 @ particle2
        print(system.state)
        print(particle1.state)
        print(particle2.state)
