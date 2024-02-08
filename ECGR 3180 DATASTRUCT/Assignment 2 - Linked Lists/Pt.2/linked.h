#ifndef LINKED
#define LINKED

struct Node {
    int data;
    Node* next;
};

class LinkedList {
public:
    LinkedList() : head(nullptr) {}
    ~LinkedList();
    void push_back(int data);
    void pop_back();
    bool remove(int index);
    bool insert(int index, int data);
    int& at(int index) const;
    int size() const;


private:
    Node* head;
};

LinkedList::~LinkedList() {
    // Destructor to clean up memory when the list is destroyed
    while (head) {
        Node* temp = head;
        head = head->next;
        delete temp;
    }
std::cout << std::endl << "Destructor Called" << std::endl;

}

int LinkedList::size() const {
    int count = 0;
    Node* currentNode = head;
    while (currentNode != nullptr) {
        count++;
        currentNode = currentNode->next;
    }
    return count;
}

void LinkedList::push_back(int data) {
    // Add a new node to the end of the list
    if (head == nullptr) {
        // If the list is empty, create the first node
        head = new Node;
        head->data = data;
        head->next = nullptr;
        return;
    }
    // Move to the last node in the list
    Node* currentNode = head;
    while (currentNode->next != nullptr) {
        currentNode = currentNode->next;
    }
    // Add the new node to the end of the list
    currentNode->next = new Node;
    currentNode->next->data = data;
    currentNode->next->next = nullptr;
}

void LinkedList::pop_back() {
    // Remove the last node from the list
    if (head == nullptr) {
        // List is empty, nothing to remove
        return;
    }
    if (head->next == nullptr) {
        // List only has one node, delete it
        delete head;
        head = nullptr;
        return;
    }
    // If there is more than one node, find the second-to-last node
    Node* currentNode = head;
    while (currentNode->next->next != nullptr) {
        currentNode = currentNode->next;
    }
    // Remove the last node and update the next pointer
    delete currentNode->next;
    currentNode->next = nullptr;
}

bool LinkedList::remove(int index) {
    if (index < 0) {
        // Handle invalid index (e.g., return false or throw an exception)
        return false;
    }
    if (index == 0) {
        // Remove the first node
        Node* temp = head->next;
        delete head;
        head = temp;
        return true;
    }
    Node* currentNode = head;
    for (int i = 0; i < index - 1; i++) {
        if (currentNode == nullptr || currentNode->next == nullptr) {
            // Handle invalid index (e.g., return false or throw an exception)
            return false;
        }
        currentNode = currentNode->next;
    }
    if (currentNode->next == nullptr) {
        // Handle invalid index (e.g., return false or throw an exception)
        return false;
    }
    Node* temp = currentNode->next->next;
    delete currentNode->next;
    currentNode->next = temp;
    return true;
}


bool LinkedList::insert(int index, int data) {
    if (index < 0) {
        // Handle invalid index (e.g., return false or throw an exception)
        return false;
    }
    if (index == 0) {
        // Insert at the beginning
        Node* temp = head;
        head = new Node;
        head->data = data;
        head->next = temp;
        return true;
    }
    Node* currentNode = head;
    for (int i = 0; i < index - 1; i++) {
        if (currentNode == nullptr) {
            // Handle invalid index (e.g., return false or throw an exception)
            return false;
        }
        currentNode = currentNode->next;
    }
    Node* temp = currentNode->next;
    currentNode->next = new Node;
    currentNode->next->data = data;
    currentNode->next->next = temp;
    return true;
}

int& LinkedList::at(int index) const {
    // Access the node at the specified index
    Node* currentNode = head;
    for (int i = 0; i < index; i++) {
        currentNode = currentNode->next;
    }
    return currentNode->data;
}

#endif
