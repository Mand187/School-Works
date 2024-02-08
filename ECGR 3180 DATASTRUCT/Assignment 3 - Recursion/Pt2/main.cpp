#include <iostream>
#include <vector>

using namespace std;

int recursiveBinarySearch(vector<int>& numArray, int target, int left, int right) {
    if (left > right) {
        // Element not found
        return -1;
    }

    int mid = left + (right - left) / 2;

    if (numArray[mid] == target) {
        // Element found at mid index
        return mid;
    }

    // Check if the target is at the leftmost or rightmost side
    if (numArray[left] == target) {
        return left;
    }
    if (numArray[right] == target) {
        return right;
    }

    if (numArray[mid] < target) {
        // Search the right subarray
        return recursiveBinarySearch(numArray, target, mid + 1, right);
    } else {
        // Search the left subarray
        return recursiveBinarySearch(numArray, target, left, mid - 1);
    }
}

int main() {
    vector<int> numArray = {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10};

    int target1 = 9;
    int result1 = recursiveBinarySearch(numArray, target1, 0, numArray.size() - 1);
    if (result1 != -1) {
        cout << "Element " << target1 << " found at index " << result1 << endl;
    } else {
        cout << "Element " << target1 << " not found" << endl;
    }

    int target2 = 1;
    int result2 = recursiveBinarySearch(numArray, target2, 0, numArray.size() - 1);
    if (result2 != -1) {
        cout << "Element " << target2 << " found at index " << result2 << endl;
    } else {
        cout << "Element " << target2 << " not found" << endl;
    }

    int target3 = 10;
    int result3 = recursiveBinarySearch(numArray, target3, 0, numArray.size() - 1);
    if (result3 != -1) {
        cout << "Element " << target3 << " found at index " << result3 << endl;
    } else {
        cout << "Element " << target3 << " not found" << endl;
    }

    int target4 = 2;
    int result4 = recursiveBinarySearch(numArray, target4, 0, numArray.size() - 1);
    if (result4 != -1) {
        cout << "Element " << target4 << " found at index " << result4 << endl;
    } else {
        cout << "Element " << target4 << " not found" << endl;
    }

    return 0;
}
