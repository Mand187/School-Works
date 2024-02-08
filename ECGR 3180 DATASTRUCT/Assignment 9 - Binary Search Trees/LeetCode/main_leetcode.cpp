#include <iostream>
#include <vector>

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
    // Function to convert a sorted array to a balanced binary search tree
    TreeNode* sortedArrayToBST(std::vector<int>& nums) {
        // Call the helper function with the entire array's range
        return sortedArrayToBST(nums, 0, nums.size() - 1);
    }

    // Helper function to build the balanced BST
    TreeNode* sortedArrayToBST(std::vector<int>& nums, int left, int right) {
        // Base case: When the left index is greater than the right index, return null
        if (left > right) {
            return nullptr;
        }

        // Calculate the middle index
        int mid = left + (right - left) / 2;

        // Create a new TreeNode with the middle value
        TreeNode* root = new TreeNode(nums[mid]);

        // Recursively build the left and right subtrees
        root->left = sortedArrayToBST(nums, left, mid - 1);
        root->right = sortedArrayToBST(nums, mid + 1, right);

        // Return the root of the constructed BST
        return root;
    }
};

// Test function
void testSortedArrayToBST() {
    Solution solution;

    // Test case 1: Even number of elements in the array
    std::vector<int> nums1 = {1, 2, 3, 4, 5, 6};
    TreeNode* result1 = solution.sortedArrayToBST(nums1);
    // You can add code to validate the tree structure here

    // Test case 2: Odd number of elements in the array
    std::vector<int> nums2 = {1, 2, 3, 4, 5};
    TreeNode* result2 = solution.sortedArrayToBST(nums2);
    // You can add code to validate the tree structure here

    // Output the results or assertions here
}

int main() {
    testSortedArrayToBST();
    return 0;
}
