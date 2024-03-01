import unittest

import unittest
import os
from Encrpyt import EncryptDecrypt

class TestEncryptDecrypt(unittest.TestCase):

    def setUp(self):
        # Create temporary input and output files for testing
        self.input_file = 'test_input.txt'
        self.output_file = 'test_output.txt'
        with open(self.input_file, 'w') as f:
            f.write('Hello, World!')

    def tearDown(self):
        # Clean up temporary files after testing
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_encryption(self):
        EncryptDecrypt(self.input_file, self.output_file, 3, True)
        with open(self.output_file, 'r') as f:
            result = f.read()
        self.assertEqual(result, 'Khoor, Zruog!')

    def test_decryption(self):
        EncryptDecrypt(self.input_file, self.output_file, 3, False)
        with open(self.output_file, 'r') as f:
            result = f.read()
        self.assertEqual(result, 'Hello, World!')

if __name__ == '__main__':
    unittest.main()
