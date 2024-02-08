#ifndef DOUBLYLINKEDLIST_H
#define DOUBLYLINKEDLIST_H

template <typename T>
class DoublyLinkedList {
private:
    struct Node {
        T data;
        Node *prev;
        Node *next;
        Node(const T &data) : data(data), prev(nullptr), next(nullptr) {}
    };

    Node *head;
    Node *tail;
    int listSize;

public:
    DoublyLinkedList() : head(nullptr), tail(nullptr), listSize(0) {}

    ~DoublyLinkedList() {
        Node *current = head;
        while (current) {
            Node *next = current->next;
            delete current;
            current = next;
        }
    }

    DoublyLinkedList(const DoublyLinkedList &other) : head(nullptr), tail(nullptr), listSize(0){
        Node *current = other.head;
        while (current){
            push(current->data);
            current = current->next;
        }
    }

    DoublyLinkedList &operator=(const DoublyLinkedList &other){
        if (this != &other) {
            clear();
            Node *current = other.head;
            while (current){
                push(current->data);
                current = current->next;
            }
        }
        return *this;
    }
    // appends a new node on the end of the list
    void push(const T &data) {
        Node *newNode = new Node(data);
        if (!head) {
            head = newNode;
            tail = newNode;
        }
        else {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        }
        listSize++;
    }
    //removes the last element of the list
    void pop() {
        if (!tail)
            return;
        Node *prevTail = tail->prev;
        delete tail;
        tail = prevTail;
        if (tail)
        {
            tail->next = nullptr;
        }
        else
        {
            head = nullptr;
        }
        listSize--;
    }
    // returns the number of elements in the list
    int size() const {
        return listSize;
    }
    // prints all elements in the list to the console
    void print() const {
        Node *current = head;
        while (current)
        {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    T &at(int idx) {
        if (idx < 0 || idx >= listSize) {
            throw std::out_of_range("Index out of range");
        }
        Node *current = head;
        for (int i = 0; i < idx; i++)
        {
            current = current->next;
        }
        return current->data;
    }

    void insert(const T &data, int pos) {
        if (pos < 0 || pos > listSize){
            throw std::out_of_range("Invalid position for insertion");
        }
        if (pos == 0){
            Node *newNode = new Node(data);
            newNode->next = head;
            if (head){
                head->prev = newNode;
            }
            else {
                tail = newNode;
            }
            head = newNode;
        }
        else if (pos == listSize) {
            
            push(data);
            
        }
        else {
            
            Node *current = head;
            for (int i = 0; i < pos - 1; i++){
                current = current->next;
            }
            Node *newNode = new Node(data);
            newNode->next = current->next;
            newNode->prev = current;
            current->next->prev = newNode;
            current->next = newNode;
        }
        listSize++;
    }

    void remove(int pos) {
        if (pos < 0 || pos >= listSize){
            throw std::out_of_range("Invalid position for removal");
        }
        if (pos == 0)
        {
            Node *newHead = head->next;
            delete head;
            head = newHead;
            if (head){
                head->prev = nullptr;
            }
            else{
                tail = nullptr;
            }
        }
        else if (pos == listSize - 1){
            pop();
        }
        else{
            Node *current = head;
            for (int i = 0; i < pos; i++){
                current = current->next;
            }
            current->prev->next = current->next;
            current->next->prev = current->prev;
            delete current;
        }
        listSize--;
    }
    
    
    void clear()
    {
        while (head){
            pop();
        }
    }
};

#endif // DOUBLYLINKEDLIST_H
