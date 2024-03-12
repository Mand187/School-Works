from LogicGate import LogicGate, NAND, NOT

# You are required to set the D Flip Flop state, because the starting state is ambigious
# Do not rely on the default configuration use Dinput

class DFlipFlop:
    def __init__(self, name):
        self.name = name

        # Initialize NAND gates
        self.nand1 = NAND(name + " NAND1")
        self.nand2 = NAND(name + " NAND2")
        self.nand3 = NAND(name + " NAND3")
        self.nand4 = NAND(name + " NAND4")
        self.notgate = NOT(name + " NOT")

        # Initialize instance variables for data and clock
        self.data = False
        self.clock = True

        # Connect NAND gates to create a D flip-flop
        self.notgate.input1 = self.data
        self.notgate.perfLogic()

        self.nand1.input1 = self.data
        self.nand1.input2 = self.clock
        self.nand1.perfLogic()

        self.nand2.input1 = self.notgate.output
        self.nand2.input2 = self.clock
        self.nand2.perfLogic()

        self.nand3.input1 = self.nand1.output
        self.nand3.input2 = self.nand4.output
        self.nand3.perfLogic()

        self.nand4.input1 = self.nand3.output
        self.nand4.input2 = self.nand2.output
        self.nand4.perfLogic()


    def dINPUT(self, clock, data):
        # Update instance variables
        self.data = data
        self.clock = clock

        # Implement logic to update the flip-flop state based on clock and data inputs
        self.notgate.input1 = data
        self.notgate.perfLogic()

        self.nand1.input1 = data
        self.nand1.input2 = clock
        self.nand1.perfLogic()

        self.nand2.input1 = self.notgate.output
        self.nand2.input2 = clock
        self.nand2.perfLogic()

        self.nand3.input1 = self.nand1.output
        self.nand3.input2 = self.nand4.output
        self.nand3.perfLogic()

        self.nand4.input1 = self.nand3.output
        self.nand4.input2 = self.nand2.output
        self.nand4.perfLogic()

        self.q = self.nand3.output
        self.qbar = self.nand4.output
        self.output = self.nand3.output #Output of D Flip Flop


