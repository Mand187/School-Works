from LogicGate import NOT, NAND
from DFlipFlop import DFlipFlop

def main():
    nand1 = NAND("NAND1")
    notGate = NOT("NOT1")

    nand1.input1 = False
    nand1.input2 = False
    nand1.perfLogic()

    print(f"{nand1.name} input, {nand1.input1}")
    print(f"{nand1.name} input, {nand1.input2}")
    print(f"{nand1.name} output, {nand1.output}")

    notGate.input1 = False
    notGate.perfLogic()

    print(f"{notGate.name} input, {notGate.input1}")
    print(f"{notGate.name}, output {notGate.output}")

    nand1.input1 = notGate.output
    nand1.input2 = False
    nand1.perfLogic()

    print(f"{nand1.name} input, {nand1.input1}")
    print(f"{nand1.name} input, {nand1.input2}")
    print(f"{nand1.name} output, {nand1.output}")

    DFF1 = DFlipFlop("DFF1")

    DFF1.dINPUT(True,False)

    print(f"{DFF1.name} output, {DFF1.output}")
    print(f"{DFF1.name} q, {DFF1.q}")
    print(f"{DFF1.name} qbar, {DFF1.qbar}")

    DFF1.dINPUT(True,True)

    print(f"{DFF1.name} output, {DFF1.output}")
    print(f"{DFF1.name} q, {DFF1.q}")
    print(f"{DFF1.name} qbar, {DFF1.qbar}")




if __name__ == "__main__":
    main()




