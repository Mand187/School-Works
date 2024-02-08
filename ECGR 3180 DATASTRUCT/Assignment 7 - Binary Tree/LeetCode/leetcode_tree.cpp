#include <iostream>
#include <algorithm>

// Struct for a binary tree node from leetcode
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if (root == nullptr) {
            return false; // Tree is empty 
        }
        
        return isMirror(root->left, root->right);
    }
    
    bool isMirror(TreeNode* leftSubtree, TreeNode* rightSubtree) {
        if (leftSubtree == nullptr && rightSubtree == nullptr) {
            return true; // both trees empty out to nullptr
        }
        if (leftSubtree == nullptr || rightSubtree == nullptr) {
            return false; // One of the trees is returns a nullptr, which implies one is shorter than the other
        }
        
        // Check if the trees have the same value and heigh height 
        return (leftSubtree->val == rightSubtree->val) && isMirror(leftSubtree->left, rightSubtree->right) && isMirror(leftSubtree->right, rightSubtree->left);
    }
};

int main() {
    // Test case 1: A symmetric tree
    TreeNode* symmetricTree = new TreeNode(1);
    symmetricTree->left = new TreeNode(2, new TreeNode(3), new TreeNode(4));
    symmetricTree->right = new TreeNode(2, new TreeNode(4), new TreeNode(3));
    Solution solution;
    std::cout << "Test case 1: ";
    if (solution.isSymmetric(symmetricTree)) {
        std::cout << "Symmetric" << std::endl;
    } else {
        std::cout << "Not Symmetric" << std::endl;
    }

    // Test case 2: An asymmetric tree
    TreeNode* asymmetricTree = new TreeNode(1);
    asymmetricTree->left = new TreeNode(2, nullptr, new TreeNode(3));
    asymmetricTree->right = new TreeNode(2, nullptr, new TreeNode(3));
    std::cout << "Test case 2: ";
    if (solution.isSymmetric(asymmetricTree)) {
        std::cout << "Symmetric" << std::endl;
    } else {
        std::cout << "Not Symmetric" << std::endl;
    }

    return 0;
}
