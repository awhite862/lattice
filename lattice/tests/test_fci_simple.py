import unittest
from lattice.hubbard import Hubbard1D
from lattice.fci import FCISimple


class TestFCISimple(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12

        # from DMRG (ITensor)
        self.ref_2_1 = -1.561552812809
        self.ref_4_1 = -3.575365620447

    def test_1d_hubbard(self):

        # L = 2, U = 1, half-filling
        hub = Hubbard1D(2, 1.0, 1.0, boundary='o')

        myfci = FCISimple(hub, 2, m_s=0)
        e, v = myfci.run()
        out = e[0]
        ref = self.ref_2_1
        err = "Expected: {} Actual: {}".format(ref, out)
        diff = abs(out - ref)
        self.assertTrue(diff < self.thresh, err)

        # L = 4, U = 1, half-filling
        hub = Hubbard1D(4, 1.0, 1.0, boundary='o')

        myfci = FCISimple(hub, 4, m_s=0)
        e, v = myfci.run()
        out = e[0]
        ref = self.ref_4_1
        err = "Expected: {} Actual: {}".format(ref, out)
        diff = abs(out - ref)
        self.assertTrue(diff < self.thresh, err)


if __name__ == '__main__':
    unittest.main()
