import unittest
from EmployeeData import EmployeeData  # Adjust the import statement based on your actual file structure

class TestEmployeeData(unittest.TestCase):
    def setUp(self):
        self.testEmployees = [
            EmployeeData("TestEmployee1", "Test", 5, 5),
            EmployeeData("TestEmployee2", "Test", 5, 10),
            EmployeeData("TestEmployee3", "Test", 10, 5),
        ]

    def test_Name(self):
        self.assertEqual(self.testEmployees[0].getName(), "TestEmployee1")

    def test_Title(self):
        self.assertEqual(self.testEmployees[0].getTitle(), "Test")

    def test_Wages(self):
        self.assertEqual(self.testEmployees[0].getWage(), 5)

    def test_Hours(self):
        self.assertEqual(self.testEmployees[0].getHours(), 5)

    def test_CalculateWages(self):
        total_wages = sum(employee.calculateWages() for employee in self.testEmployees)
        self.assertAlmostEqual(total_wages, 125, places=2)  # Adjust the expected value as needed

if __name__ == '__main__':
    unittest.main()
