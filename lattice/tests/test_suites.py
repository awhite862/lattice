import sys
import unittest
import logging
import test_anderson
import test_hubbard
import test_test
import test_fci_simple


def run_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_anderson.AndersonTest("test_vs_hubbard_simple"))
    suite.addTest(test_anderson.AndersonTest("test_vs_hubbard"))

    suite.addTest(test_hubbard.HubbardTest("testT1D"))
    suite.addTest(test_hubbard.HubbardTest("testT2D"))
    suite.addTest(test_hubbard.HubbardTest("testT3D"))
    suite.addTest(test_hubbard.HubbardTest("testUNorm"))

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_fci_simple.TestFCISimple("test_1d_hubbard"))
    suite.addTest(test_fci_simple.TestFCISimple("test_basis_2site"))

    return suite


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.ERROR,
        stream=sys.stdout)
    runner = unittest.TextTestRunner()
    runner.run(run_suite())
