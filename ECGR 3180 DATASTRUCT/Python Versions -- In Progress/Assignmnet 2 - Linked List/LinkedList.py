class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def __del__(self):
        # Destructor to clean up memory when the list is destroyed
        current = self.head
        while current:
            temp = current
            current = current.next
            del temp
        print("\nDestructor Called")

    def size(self):
        # Get the size of the linked list
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def push_back(self, data):
        # Add a new node to the end of the list
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def pop_back(self):
        # Remove the last node from the list
        if not self.head:
            return
        if not self.head.next:
            self.head = None
            return

        current = self.head
        while current.next and current.next.next:
            current = current.next
        current.next = None

    def remove(self, index):
        # Remove the node at the specified index
        if index < 0 or not self.head:
            return False

        if index == 0:
            temp = self.head.next
            del self.head
            self.head = temp
            return True

        current = self.head
        for _ in range(index - 1):
            if not current or not current.next:
                return False
            current = current.next

        if not current.next:
            return False

        temp = current.next.next
        del current.next
        current.next = temp
        return True

    def insert(self, index, data):
        # Insert a new node at the specified index
        if index < 0:
            return False

        new_node = Node(data)
        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return True

        current = self.head
        for _ in range(index - 1):
            if not current:
                return False
            current = current.next

        new_node.next = current.next
        current.next = new_node
        return True

    def at(self, index):
        # Access the node at the specified index
        if index < 0:
            raise IndexError("Index out of range")

        current = self.head
        for _ in range(index):
            if not current:
                raise IndexError("Index out of range")
            current = current.next

        if not current:
            raise IndexError("Index out of range")

        return current.data

    def __str__(self):
        # String representation of the linked list
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return str(elements)
