from DoudblyLinkedList import DoublyLinkedList

def test():
    dll = DoublyLinkedList()

    dll.push(1)
    dll.push(2)
    dll.push(3)

    dll.print()

    dllCopy = DoublyLinkedList()
    dllCopy.__dict__ = dll.__dict__.copy()  # Corrected this line

    dllCopy.push(4)
    dllCopy.insert(5, 1)

    dll.__dict__ = dllCopy.__dict__.copy()  # Corrected this line

    dll.print()

def main():
    test()

if __name__ == "__main__":  # Corrected this line
    main()
