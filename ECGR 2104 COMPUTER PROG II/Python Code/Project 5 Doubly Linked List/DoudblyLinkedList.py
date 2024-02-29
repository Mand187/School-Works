class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def copyFrom(self, other):
        for data in other:
            self.push(data)

    def clear(self):
        while self.head:
            temp = self.head
            self.head = self.head.next
            del temp
        self.tail = None

    def push(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def pop(self):
        if not self.head:
            print("The list is empty")
            return

        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            del self.tail.next
            self.tail.next = None

    def size(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def print(self):
        current = self.head
        while current:
            print(current.data, end=" ")
            current = current.next
        print()

    def at(self, idx):
        if idx < 0 or idx >= self.size():
            print("Index out of range")
            raise IndexError("INDEX_OUT_OF_RANGE")

        current = self.head
        for i in range(idx):
            current = current.next
        return current.data

    def insert(self, data, pos):
        if pos < 0 or pos > self.size():
            raise IndexError("INVALID_INDEX_CANNOT_INSERT")

        if pos == self.size():
            self.push(data)
            return

        new_node = Node(data)
        if pos == 0:
            new_node.next = self.head
            if self.head:
                self.head.prev = new_node
            self.head = new_node
        else:
            current = self.head
            for i in range(pos - 1):
                current = current.next
            new_node.next = current.next
            new_node.prev = current
            current.next.prev = new_node
            current.next = new_node

    def remove(self, pos):
        if pos < 0 or pos >= self.size():
            raise IndexError("INVALID_INDEX_CANNOT_REMOVE")

        if pos == self.size() - 1:
            self.pop()

        elif pos == 0:
            temp = self.head
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
            del temp

        else:
            current = self.head
            for i in range(pos):
                current = current.next
            current.prev.next = current.next
            current.next.prev = current.prev
            del current
