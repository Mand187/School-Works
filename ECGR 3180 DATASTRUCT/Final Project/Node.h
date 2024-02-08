// Min Chang
// Github: Minyc510

#ifndef NODE_H
#define NODE_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>

using namespace std;

class Node {

private:
  double data;  // Data associated with the node
  string name;  // Unique name of the node

  // neighborMap: List of Nodes that this node has an edge to, along with corresponding edge weights
  unordered_map<string, multiset<double>>* neighborMap;

  // neighborOfSet: List of Nodes that have an edge to this Node
  unordered_set<string> neighborOfSet;

public:
  // Constructor
  Node(double data, string name);

  // Destructor
  ~Node();

  // Method to add a neighbor with a given name and edge weight
  void addNeighbor(string neighborName, double weight);

  // Accessor methods

  // Get the data associated with the node
  double getData();

  // Get a pointer to the neighborMap
  unordered_map<string, multiset<double>>* getMapPtr();

  // Get a reference to the neighborOfSet
  unordered_set<string>& getSetRef();
};

#endif // NODE_H
