// Min Chang
// Github: Minyc510

#include "Node.h"
#include <utility>

using namespace std;

// Constructor: Initializes the node with provided data and name
Node::Node(double data, string name) {
  this->data = data;
  this->name = name;
  
  // Dynamically allocate neighborMap
  unordered_map<string, multiset<double>>* mapPointer = new unordered_map<string, multiset<double>>();
  neighborMap = mapPointer;
}

// Destructor: Deallocates memory used by neighborMap
Node::~Node() {
  delete neighborMap;
}

// Method to add a neighbor with a given name and edge weight
void Node::addNeighbor(string neighborName, double weight) {
  // If the new neighbor is not already a neighbor, add it to the list
  if (neighborMap->find(neighborName) == neighborMap->end()) {
    multiset<double> tempSet;
    pair<string, multiset<double>> tempPair(neighborName, tempSet);
    neighborMap->insert(tempPair);
  }

  // Add an edge of this 'weight'
  (*neighborMap)[neighborName].insert(weight);
}

// Accessor method: Get the data associated with the node
double Node::getData() {
  return data;
}

// Accessor method: Get a pointer to the neighborMap
unordered_map<string, multiset<double>>* Node::getMapPtr() {
  return neighborMap;
}

// Accessor method: Get a reference to the neighborOfSet
unordered_set<string>& Node::getSetRef() {
  return neighborOfSet;
}
