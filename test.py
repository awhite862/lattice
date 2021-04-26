import logging
import sys
from lattice.tests.test_test import *
from lattice.tests.test_anderson import *
from lattice.tests.test_fci_simple import *

logging.basicConfig(
    format='%(levelname)s:%(message)s',
    level=logging.ERROR,
    stream=sys.stdout)
