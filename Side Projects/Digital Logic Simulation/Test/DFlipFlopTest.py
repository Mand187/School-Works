import unittest
from DFlipFlop import DFlipFlop

class TestDFlipFlop(unittest.TestCase):
    def test_d_flip_flop(self):
        # Initialize DFlipFlop instance
        dff = DFlipFlop("TestDFF")

        # Test with clock = True, data = False
        dff.dINPUT(clock=True, data=False)
        self.assertEqual(dff.output, False)
        self.assertEqual(dff.q, False)
        self.assertEqual(dff.qbar, True)

        # Test with clock = True, data = True
        dff.dINPUT(clock=True, data=True)
        self.assertEqual(dff.output, True)
        self.assertEqual(dff.q, True)
        self.assertEqual(dff.qbar, False)

        # Test with clock = False, data = True
        dff.dINPUT(clock=False, data=True)
        self.assertEqual(dff.output, True)
        self.assertEqual(dff.q, True)
        self.assertEqual(dff.qbar, False)

if __name__ == '__main__':
    unittest.main()
