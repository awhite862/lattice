import unittest
import numpy
from lattice.hubbard import Hubbard1D, Hubbard2D, Hubbard3D


class HubbardTest(unittest.TestCase):
    def testT1D(self):
        # 2 sites, open boudnary
        hub = Hubbard1D(2, 1.0, 0.0, boundary='o')
        tref = numpy.zeros((2, 2))
        tref[0, 1] = -1
        tref[1, 0] = -1
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)

        # 2 sites, periodic boundary
        hub = Hubbard1D(2, 1.0, 0.0, boundary='p')
        tref = numpy.zeros((2,2))
        tref[0,1] = -2
        tref[1,0] = -2
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)

        # 3 sites, periodic boundary
        hub = Hubbard1D(3, 1.0, 0.0, boundary='p')
        tref = numpy.zeros((3, 3))
        tref[0, 1] = -1
        tref[1, 0] = -1
        tref[1, 2] = -1
        tref[2, 1] = -1
        tref[0, 2] = -1
        tref[2, 0] = -1
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)

    def testT2D(self):
        # 4 sites open boundary
        nn = [(1, 2), (0, 3), (0, 3), (2, 1)]
        hub = Hubbard2D(4, 1.0, 0.0, nn)
        tref = numpy.zeros((4, 4))
        tref[0, 1] = -1
        tref[1, 0] = -1
        tref[2, 0] = -1
        tref[0, 2] = -1
        tref[3, 1] = -1
        tref[1, 3] = -1
        tref[3, 2] = -1
        tref[2, 3] = -1
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)

        # 9 sites periodic boundary
        nn = [(1, 3, 2, 6), (0, 7, 2, 4), (1, 8, 0, 5),
              (5, 0, 4, 6), (3, 1, 5, 7), (4, 2, 3, 8),
              (8, 3, 7, 0), (6, 4, 8, 1), (7, 5, 6, 2)]
        hub = Hubbard2D(9, 1.0, 0.0, nn)
        tref = numpy.zeros((9, 9))
        tref[0, 2] = -1
        tref[0, 6] = -1
        tref[0, 1] = -1
        tref[0, 3] = -1
        tref[1, 7] = -1
        tref[1, 2] = -1
        tref[1, 4] = -1
        tref[2, 8] = -1
        tref[2, 5] = -1
        tref[3, 5] = -1
        tref[3, 4] = -1
        tref[3, 6] = -1
        tref[4, 5] = -1
        tref[4, 7] = -1
        tref[5, 8] = -1
        tref[6, 7] = -1
        tref[6, 8] = -1
        tref[7, 8] = -1
        tref = tref + tref.transpose()
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)

    def testT3D(self):
        # 8 sites  boundary
        nn = [(1, 2, 4), (0, 3, 5), (0, 6, 3), (1, 2, 7),
              (0, 6, 5), (4, 7, 1), (2, 4, 7), (3, 5, 6)]
        hub = Hubbard3D(8, 1.0, 0.0, nn)
        tref = numpy.zeros((8, 8))
        tref[0, 1] = -1
        tref[0, 2] = -1
        tref[0, 4] = -1
        tref[1, 3] = -1
        tref[1, 5] = -1
        tref[2, 3] = -1
        tref[2, 6] = -1
        tref[3, 7] = -1
        tref[4, 6] = -1
        tref[4, 5] = -1
        tref[5, 7] = -1
        tref[6, 7] = -1
        tref = tref + tref.transpose()
        tout = hub.get_tmatS()
        diff = numpy.linalg.norm(tout - tref)
        self.assertTrue(diff < 1e-14)


if __name__ == '__main__':
    unittest.main()
