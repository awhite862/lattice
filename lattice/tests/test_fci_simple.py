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

    def test_basis_2site(self):
        hub = Hubbard1D(2, 1.0, 1.0, boundary='o')
        myfci = FCISimple(hub, 2)
        out = myfci.print_basis()

        ref = "|0 1> m_s = 2\n"
        ref += "|0 2> m_s = 0\n"
        ref += "|0 3> m_s = 0\n"
        ref += "|1 2> m_s = 0\n"
        ref += "|1 3> m_s = 0\n"
        ref += "|2 3> m_s = -2\n"
        self.assertTrue(out == ref)


if __name__ == '__main__':
    unittest.main()
