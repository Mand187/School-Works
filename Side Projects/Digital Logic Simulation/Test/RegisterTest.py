import unittest
from Register import Register

class TestRegister(unittest.TestCase):
    def test_register(self):
        # Create a register with 8 bits
        my_register = Register("my_register", 8)
        
        # Set input data
        input_data = [True, False, True, False, True, False, True, False]
        my_register.setInput(input_data)
        
        # Read the output data
        output_data = my_register.readOutput()
        
        # Check if output data matches input data
        self.assertEqual(output_data, input_data)

if __name__ == '__main__':
    unittest.main()
