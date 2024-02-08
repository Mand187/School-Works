/* Given: A sequence of numbers x1, x2, ..., xn input one-by-one, find the median as the numbers arrive.
 * The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
 * Your algorithms should be O(nlogn) where n is the number of elements seen thus far.
 * Hint: Solution involves using a max and min heap (STL priority queues) discussed in class.
 */

#include <iostream>
#include <queue>
#include <vector>

const int MAX_VAL = 100;
const int NUM_ELEM = 15;

template <typename T>
void printQueue(T &q) {
    T q_copy(q);
    while (!q_copy.empty()) {
        std::cout << q_copy.top() << " ";
        q_copy.pop();
    }
    std::cout << std::endl;
}

std::vector<double> findMedian(std::vector<int> &data) {
    std::priority_queue<double> max_heap;
    std::priority_queue<double, std::vector<double>, std::greater<double>> min_heap; // Changed to dobule to match expecte result

    // Initialize a vector to store the calculated medians.
    std::vector<double> resultVec;

    // Initialize a variable to hold the current median
    double median = 0;

    // Iterate through each element in the data stream.
    for (int i = 0; i < data.size(); ++i) {
        double x = data[i];

        // If the max-heap is empty or the current element is less than or equal to the maximum element in the max-heap,
        if (max_heap.empty() || x <= max_heap.top()) {
            // Insert the element into the max-heap.
            max_heap.push(x);
        } else {
            // Otherwise, insert the element into the min-heap.
            min_heap.push(x);
        }

        // Ensure that the sizes of the max-heap and min-heap differ by at most 1.
        if (max_heap.size() > min_heap.size() + 1) {
            // Move an element from the max-heap to the min-heap to maintain balance.
            min_heap.push(max_heap.top());
            max_heap.pop();
        } else if (min_heap.size() > max_heap.size()) {
            // Move an element from the min-heap to the max-heap to maintain balance.
            max_heap.push(min_heap.top());
            min_heap.pop();
        }

        // Calculate the current median:
        if (max_heap.size() == min_heap.size()) {
            // If both heaps have the same size, the median is the average of the top elements of both heaps.
            median = (max_heap.top() + min_heap.top()) / 2.0;
        } else {
            // If the sizes are not equal, the median is the top element of the max-heap.
            median = max_heap.top();
        }

        // Add the current median to the result vector.
        resultVec.push_back(median);
    }

    // Return the vector containing the calculated medians.
    return resultVec;
}


int main() {
    std::vector<int> data_stream = {5, 42, 29, 85, 95, 99, 2, 15};
    std::vector<double> median_stream = findMedian(data_stream);
    for (auto ele : median_stream) {
        std::cout << ele << " "; // Answer should be: 5 23.5 29 35.5 42 63.5 42 35.5
    }
    std::cout << std::endl;
}
