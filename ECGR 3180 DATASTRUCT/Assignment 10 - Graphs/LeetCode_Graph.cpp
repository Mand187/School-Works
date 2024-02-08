#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        // Create an adjacency list to represent the graph
        vector<unordered_set<int>> graph(n);
        for (const auto& edge : edges) {
            graph[edge[0]].insert(edge[1]);
            graph[edge[1]].insert(edge[0]);
        }

        vector<bool> visited(n, false);
        return dfs(graph, visited, source, destination);
    }

private:
    bool dfs(const vector<unordered_set<int>>& graph, vector<bool>& visited, int current, int destination) {
        if (current == destination) {
            return true;  // Found a valid path
        }

        visited[current] = true;

        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                if (dfs(graph, visited, neighbor, destination)) {
                    return true;  // Found a valid path
                }
            }
        }

        return false;  // No valid path found
    }
};

int main() {
    Solution solution;

    // Example 1
    int n1 = 3;
    vector<vector<int>> edges1 = {{0, 1}, {1, 2}, {2, 0}};
    int source1 = 0, destination1 = 2;
    cout << "Example 1: " << solution.validPath(n1, edges1, source1, destination1) << endl;  // Output: true

    // Example 2
    int n2 = 6;
    vector<vector<int>> edges2 = {{0, 1}, {0, 2}, {3, 5}, {5, 4}, {4, 3}};
    int source2 = 0, destination2 = 5;
    cout << "Example 2: " << solution.validPath(n2, edges2, source2, destination2) << endl;  // Output: false

    return 0;
}
