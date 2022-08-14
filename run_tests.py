
import unittest

# import test modules

from containers.tests import test_LINEAR_REGRESSION
from containers.tests import test_LOGISTIC_REGRESSION
from containers.tests import test_LASSO_REGRESSION
from containers.tests import test_DECISION_TREE

# initialize test suite

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite

suite.addTest(loader.loadTestsFromModule(test_LINEAR_REGRESSION))
suite.addTest(loader.loadTestsFromModule(test_LOGISTIC_REGRESSION))
suite.addTest(loader.loadTestsFromModule(test_LASSO_REGRESSION))
suite.addTest(loader.loadTestsFromModule(test_DECISION_TREE))


# initialize a test runner and run the test suite

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
