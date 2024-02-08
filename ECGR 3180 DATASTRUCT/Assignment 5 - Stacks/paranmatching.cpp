/*
Write a function that returns if a string of paranthesis are matched.
You are required to use the STL stack datastructure in your code for O(n) time
complexity.
See 5.17 in the book for problem description and a stack based algorithm.
*/
#include <iostream>
#include <stack>
#include <vector>

using namespace std;

// Function to check if parentheses are balanced
bool areParenthesesBalanced(string expr) {
    stack<char> stack; // Create a stack to store opening parentheses

    // Loop through each character in the input string
    for (char charData : expr) {
        if (charData == '(' || charData == '[' || charData == '{') {
            // If an opening parenthesis is encountered, push it onto the stack
            stack.push(charData);
        } else if (charData == ')' || charData == ']' || charData == '}') {
            // If a closing parenthesis is encountered
            if (stack.empty()) {
                return false; // There is a closing parenthesis without a corresponding opening parenthesis.
            }
            char top = stack.top(); // Get the top of the stack
            stack.pop(); // Pop the top element from the stack

            // Check if the current closing parenthesis matches the top of the stack
            if ((charData == ')' && top != '(') || (charData == ']' && top != '[') || (charData == '}' && top != '{')) {
                return false; // Mismatched opening and closing parenthesis.
            }
        }
    }

    // If the stack is empty at the end, all parentheses are matched.
    return stack.empty();
}

int main() {
    vector<string> parans = {"{()}[]", "[[", "))", "{[()]}", "({[])}"};

    for (auto expr : parans) {
        if (areParenthesesBalanced(expr))
            cout << "Balanced" << endl;
        else
            cout << "Not Balanced" << endl;
    }
    return 0;
}
