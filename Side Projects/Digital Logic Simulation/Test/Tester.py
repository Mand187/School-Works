import unittest
from Test.DFlipFlopTest import TestDFlipFlop
from Test.LogicGateTest import TestLogicGate
from Test.RegisterTest import TestRegister

if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()

    # Add the test cases
    suite.addTest(unittest.makeSuite(TestDFlipFlop))
    suite.addTest(unittest.makeSuite(TestLogicGate))
    suite.addTest(unittest.makeSuite(TestRegister))

    # Run the test suite
    unittest.TextTestRunner().run(suite)
