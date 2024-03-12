class LogicGate:
    def __init__(self, name):
        self.input1 = False
        self.input2 = False
        self.output = False
        self.name = name
    
    def perfLogic(self):
        pass

class NAND(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = not (self.input1 and self.input2)

class AND(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = self.input1 and self.input2 

class OR(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = self.input1 or self.input2

class NOT(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = not self.input1

class NOR(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = not (self.input1 and self.input2)

class XOR(LogicGate):
    def __init__(self, name):
        super().__init__(name)

    def perfLogic(self):
        self.output = self.input1 != self.input2
