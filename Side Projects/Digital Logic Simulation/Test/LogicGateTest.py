import unittest
from LogicGate import NAND, AND, OR, NOT, NOR, XOR

class TestLogicGate(unittest.TestCase):
    def setUp(self):
        # Create an instance of each LogicGate for each test
        self.nand_gate = NAND("NAND_Gate")
        self.and_gate = AND("AND_Gate")
        self.or_gate = OR("OR_Gate")
        self.not_gate = NOT("NOT_Gate")
        self.nor_gate = NOR("NOR_Gate")
        self.xor_gate = XOR("XOR_Gate")

    def test_NAND_gate(self):
        self.assertEqual(self.nand_gate.name, "NAND_Gate")

        self.nand_gate.input1 = False
        self.nand_gate.input2 = False
        self.nand_gate.perfLogic()
        self.assertEqual(self.nand_gate.output, True)

        self.nand_gate.input1 = False
        self.nand_gate.input2 = True
        self.nand_gate.perfLogic()
        self.assertEqual(self.nand_gate.output, True)

        # Add similar tests for other cases

    def test_AND_gate(self):
        self.assertEqual(self.and_gate.name, "AND_Gate")

        self.and_gate.input1 = False
        self.and_gate.input2 = False
        self.and_gate.perfLogic()
        self.assertEqual(self.and_gate.output, False)

        # Add similar tests for other cases

    def test_OR_gate(self):
        self.assertEqual(self.or_gate.name, "OR_Gate")

        self.or_gate.input1 = False
        self.or_gate.input2 = False
        self.or_gate.perfLogic()
        self.assertEqual(self.or_gate.output, False)

        # Add similar tests for other cases

    def test_NOT_gate(self):
        self.assertEqual(self.not_gate.name, "NOT_Gate")

        self.not_gate.input1 = False
        self.not_gate.perfLogic()
        self.assertEqual(self.not_gate.output, True)

        # Add similar tests for other cases

    def test_NOR_gate(self):
        self.assertEqual(self.nor_gate.name, "NOR_Gate")

        self.nor_gate.input1 = False
        self.nor_gate.input2 = False
        self.nor_gate.perfLogic()
        self.assertEqual(self.nor_gate.output, True)

        # Add similar tests for other cases

    def test_XOR_gate(self):
        self.assertEqual(self.xor_gate.name, "XOR_Gate")

        self.xor_gate.input1 = False
        self.xor_gate.input2 = False
        self.xor_gate.perfLogic()
        self.assertEqual(self.xor_gate.output, False)

        # Add similar tests for other cases

if __name__ == '__main__':
    unittest.main()
