import unittest
from DoudblyLinkedList import DoublyLinkedList

class TestDoublyLinkedList(unittest.TestCase):

    def setUp(self):
        # Create a new DoublyLinkedList for each test case
        self.dll = DoublyLinkedList()

    def test_push_pop(self):
        self.dll.push(1)
        self.assertEqual(self.dll.size(), 1)
        self.dll.push(2)
        self.dll.push(3)
        self.assertEqual(self.dll.size(), 3)

        self.dll.pop()
        self.assertEqual(self.dll.size(), 2)
        self.dll.pop()
        self.assertEqual(self.dll.size(), 1)
        self.dll.pop()
        self.assertEqual(self.dll.size(), 0)

    def test_insert_remove(self):
        self.dll.insert(1, 0)
        self.assertEqual(self.dll.size(), 1)
        self.dll.insert(2, 0)
        self.dll.insert(3, 1)
        self.assertEqual(self.dll.size(), 3)

        self.dll.remove(1)
        self.assertEqual(self.dll.size(), 2)
        self.dll.remove(0)
        self.assertEqual(self.dll.size(), 1)
        self.dll.remove(0)
        self.assertEqual(self.dll.size(), 0)

    def test_copyFrom(self):
        other_dll = DoublyLinkedList()
        other_dll.push(1)
        other_dll.push(2)
        other_dll.push(3)

        self.dll.copyFrom(other_dll)

        self.assertEqual(self.dll.size(), 3)
        self.assertEqual(self.dll.at(0), 1)
        self.assertEqual(self.dll.at(1), 2)
        self.assertEqual(self.dll.at(2), 3)

    def test_clear(self):
        self.dll.push(1)
        self.dll.push(2)
        self.dll.push(3)

        self.dll.clear()

        self.assertEqual(self.dll.size(), 0)

    def test_at_invalid_index(self):
        with self.assertRaises(IndexError):
            self.dll.at(0)

    def test_insert_invalid_index(self):
        with self.assertRaises(IndexError):
            self.dll.insert(1, 1)

    def test_remove_invalid_index(self):
        with self.assertRaises(IndexError):
            self.dll.remove(0)

if __name__ == '__main__':
    unittest.main()
