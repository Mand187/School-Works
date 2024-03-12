from DFlipFlop import DFlipFlop

class Register:
    def __init__(self, name, bits):
        self.name = name
        self.bits = bits
        self.flip_flops = [DFlipFlop(f"{name}_bit_{i}") for i in range(bits)]
        self.data = [False] * bits

    def setInput(self, data):
        if len(data) != self.bits:
            raise ValueError("Input data size does not match the number of bits in the register.")
        
        for i, bit_data in enumerate(data):
            self.flip_flops[i].dINPUT(True, bit_data)
            self.data[i] = bit_data

    def readOutput(self):
        return self.data
