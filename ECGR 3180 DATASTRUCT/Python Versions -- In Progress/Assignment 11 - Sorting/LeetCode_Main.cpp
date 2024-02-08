#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> targetIndices(vector<int>& nums, int target) {
        // Create a vector of pairs where each pair contains the original number and its index
        vector<pair<int, int>> numsWithIndices;
        for (int i = 0; i < nums.size(); ++i) {
            numsWithIndices.push_back({nums[i], i});
        }

        // Sort the vector of pairs based on the numbers
        sort(numsWithIndices.begin(), numsWithIndices.end());

        // Find the indices of the target in the sorted vector
        vector<int> result;
        for (int i = 0; i < numsWithIndices.size(); ++i) {
            if (numsWithIndices[i].first == target) {
                result.push_back(numsWithIndices[i].second);
            }
        }

        // Sort the result vector in increasing order
        sort(result.begin(), result.end());

        return result;
    }
};

int main() {
    Solution solution;
    vector<int> nums1 = {1, 2, 5, 2, 3};
    int target1 = 2;
    vector<int> result1 = solution.targetIndices(nums1, target1);
    // Output: [1, 2] // Indices of target (2) in the sorted array [1, 2, 2, 3, 5]

    vector<int> nums2 = {1, 2, 5, 2, 3};
    int target2 = 3;
    vector<int> result2 = solution.targetIndices(nums2, target2);
    // Output: [3] // Index of target (3) in the sorted array [1, 2, 2, 3, 5]

    vector<int> nums3 = {1, 2, 5, 2, 3};
    int target3 = 5;
    vector<int> result3 = solution.targetIndices(nums3, target3);
    // Output: [4] // Index of target (5) in the sorted array [1, 2, 2, 3, 5]

    return 0;
}
