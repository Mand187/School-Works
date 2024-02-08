#include <iostream>
#include <queue>

using namespace std;

class MyStack {
private:
    queue<int> q1; // Primary queue representing the stack
    queue<int> q2; // Temporary queue for push operation

public:
    MyStack() {
        // Default Constructor
    }

    void push(int x) {
        // Push the element onto the stack
        // Move all elements from the primary queue (q1) to the temporary queue (q2)
        while (!q1.empty()) {
            q2.push(q1.front()); // Move elements from q1 to q2
            q1.pop();
        }

        // Add the new element to the now empty primary queue (q1)
        q1.push(x);

        // Move elements back from the temporary queue (q2) to the primary queue (q1)
        while (!q2.empty()) {
            q1.push(q2.front()); // Move elements from q2 to q1
            q2.pop();
        }
    }

    int pop() {
        if (!empty()) {
            // Pop the top element from the stack
            int topElement = q1.front();
            q1.pop();
            return topElement;
        }
        return 0; // Stack is empty
    }

    int top() {
        if (!empty()) {
            // Return the top element from the stack
            return q1.front();
        }
        return 0; // Stack is empty
    }

    bool empty() {
        // Check if the stack is empty
        return q1.empty();
    }
};

int main() {
    MyStack myStack;
    myStack.push(1);
    myStack.push(2);
    myStack.push(3);

    cout << "Top: " << myStack.top() << endl; 
    cout << "Pop: " << myStack.pop() << endl; 
    cout << "Empty: " << myStack.empty() << endl; // Output: 0 (false)

    return 0;
}
