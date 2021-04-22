import numpy
import unittest
from lattice.hubbard import Hubbard1D
from lattice.anderson import Anderson


class AndersonTest(unittest.TestCase):

    def test_vs_hubbard_simple(self):
        # L = 5, U = 0, half-filling
        hub = Hubbard1D(5, 1.0, 0.0, boundary='c')
        aim = Anderson(2, 2, 1.0, 1.0, 0.0, 1.0, 0.0)

        tsref = hub.get_tmatS()
        tsout = aim.get_tmatS()

        self.assertTrue(numpy.linalg.norm(tsref - tsout) < 1e-14)

        tref = hub.get_tmat()
        tout = aim.get_tmat()

        self.assertTrue(numpy.linalg.norm(tref - tout) < 1e-14)

    def test_vs_hubbard(self):
        # L = 5, U = 0, half-filling
        hub = Hubbard1D(5, 1.0, 0.0, boundary='c')
        aim = Anderson(2, 2, 1.0, 2.0, 0.0, 1.0, 1.0)

        tsref = hub.get_tmatS()
        tsout = aim.get_tmatS()
        tsref[2,3] = -2
        tsref[3,2] = -2
        tsref[2,1] = -2
        tsref[1,2] = -2

        self.assertTrue(numpy.linalg.norm(tsref - tsout) < 1e-14)

        tref = hub.get_tmat()
        tout = aim.get_tmat()
        tref[2,3] = -2
        tref[3,2] = -2
        tref[1,2] = -2
        tref[2,1] = -2
        tref[2 + 5,3 + 5] = -2
        tref[3 + 5,2 + 5] = -2
        tref[1 + 5,2 + 5] = -2
        tref[2 + 5,1 + 5] = -2

        self.assertTrue(numpy.linalg.norm(tref - tout) < 1e-14)


if __name__ == '__main__':
    unittest.main()
