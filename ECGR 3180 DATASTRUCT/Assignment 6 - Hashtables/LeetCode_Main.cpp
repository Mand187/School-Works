#include <iostream>

// Define a struct for a linked list node
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Function to check if a linked list has a cycle
bool hasCycle(ListNode* head) {
    if (head == nullptr || head->next == nullptr) {
        return false;
    }

    ListNode* rearPointer = head;
    ListNode* forwardPointer = head->next;

    while (rearPointer != forwardPointer) {
        if (forwardPointer == nullptr || forwardPointer->next == nullptr) {
            return false;
        }
        rearPointer = rearPointer->next;
        forwardPointer = forwardPointer->next->next;
    }

    return true;
}

int main() {
    // Create a sample linked list with a cycle
    ListNode* head = new ListNode(3);
    ListNode* node1 = new ListNode(2);
    ListNode* node2 = new ListNode(0);
    ListNode* node3 = new ListNode(-4);

    head->next = node1;
    node1->next = node2;
    node2->next = node3;
    node3->next = node1;  // Create a cycle

    // Check if the linked list has a cycle
    if (hasCycle(head)) {
        std::cout << "The linked list has a cycle." << std::endl;
    } else {
        std::cout << "The linked list does not have a cycle." << std::endl;
    }

    return 0;
}
