#include "DoublyLinkedList.h"
#include <iostream>

// Constructor
template <typename T>
DoublyLinkedList<T>::DoublyLinkedList() : head(nullptr), tail(nullptr) {}

// Copy constructor
template <typename T>
DoublyLinkedList<T>::DoublyLinkedList(const DoublyLinkedList& other) : head(nullptr), tail(nullptr) {
    copyFrom(other);
}

// Destructor
template <typename T>
DoublyLinkedList<T>::~DoublyLinkedList() {
    clear();
}

// Private helper function for deep copy
template <typename T>
void DoublyLinkedList<T>::copyFrom(const DoublyLinkedList& other) {
    Node* current = other.head;
    while (current) {
        push(current->data);
        current = current->next;
    }
}

// Private helper function for garbage collection
template <typename T>
void DoublyLinkedList<T>::clear() {
    while (head) {
        Node* temp = head;
        head = head->next;
        delete temp;
    }
    tail = nullptr;
}

// Push a new element to the end of the list
template <typename T>
void DoublyLinkedList<T>::push(T data) {
    Node* newNode = new Node(data);
    if (!head) {
        head = tail = newNode;
    } else {
        tail->next = newNode;
        newNode->prev = tail;
        tail = newNode;
    }
}

// Remove the last element from the list
template <typename T>
void DoublyLinkedList<T>::pop() {
    if (!head) {
        std::cout << "The list is empty. Cannot pop." << std::endl;
        return;
    }
    if (head == tail) {
        delete head;
        head = tail = nullptr;
    } else {
        tail = tail->prev;
        delete tail->next;
        tail->next = nullptr;
    }
}

// Returns the number of elements in the list
template <typename T>
int DoublyLinkedList<T>::size() const {
    int count = 0;
    Node* current = head;
    while (current) {
        count++;
        current = current->next;
    }
    return count;
}

// Print the elements of the list
template <typename T>
void DoublyLinkedList<T>::print() const {
    Node* current = head;
    while (current) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;
}

// Returns a reference to the data at the requested index
template <typename T>
T& DoublyLinkedList<T>::at(int idx) {
    if (idx < 0 || idx >= size()) {
        std::cout << "Index out of range." << std::endl;
        throw std::out_of_range("Index out of range.");
    }

    Node* current = head;
    int i = 0;
    while (current && i < idx) {
        current = current->next;
        i++;
    }
    return current->data;
}

// Insert a new node containing data at the position "pos" in the list
template <typename T>
void DoublyLinkedList<T>::insert(T data, int pos) {
    if (pos < 0 || pos > size()) {
        std::cout << "Invalid position. Cannot insert." << std::endl;
        return;
    }

    if (pos == size()) {
        push(data);
        return;
    }

    if (pos == 0) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head->prev = newNode;
        head = newNode;
    } else {
        Node* current = head;
        int i = 0;
        while (current && i < pos) {
            current = current->next;
            i++;
        }
        if (current) {
            Node* newNode = new Node(data);
            newNode->next = current;
            newNode->prev = current->prev;
            current->prev->next = newNode;
            current->prev = newNode;
        } else {
            std::cout << "Invalid position. Cannot insert." << std::endl;
        }
    }
}

// Remove a node at the specified index
template <typename T>
void DoublyLinkedList<T>::remove(int pos) {
    if (pos < 0 || pos >= size()) {
        std::cout << "Invalid position. Cannot remove." << std::endl;
        return;
    }

    if (pos == size() - 1) {
        pop();
    } else if (pos == 0) {
        Node* temp = head;
        head = head->next;
        head->prev = nullptr;
        delete temp;
    } else {
        Node* current = head;
        int i = 0;
        while (current && i < pos) {
            current = current->next;
            i++;
        }
        if (current) {
            current->prev->next = current->next;
            current->next->prev = current->prev;
            delete current;
        } else {
            std::cout << "Invalid position. Cannot remove." << std::endl;
        }
    }
}
