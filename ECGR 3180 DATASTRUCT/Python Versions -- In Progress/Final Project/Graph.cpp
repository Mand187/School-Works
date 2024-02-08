//Min Chang
//Github: Minyc510

#include "Graph.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits> //Simulate infinity
#include <queue>

using namespace std;

Graph::Graph() {}

Graph::Graph(bool directed) { this->directed = directed; }

///Copy Constructor
Graph::Graph(const Graph& original) {
  //Copy over boolean's
  directed = original.directed;

  //Add all nodes in original to new Graph
  for (auto iter : original.nodeMap) {
    int data = iter.second->getData();
    string name = iter.first;

    Node* newNode = new Node(data, name);
    nodeMap.emplace(name, newNode);
  }

  //Add all edges in original to new Graph
  vector< tuple<string, string, double> > edgeVec = original.getEdges();
  for (auto edge : edgeVec) {
    string nodeA = get<0>(edge);
    string nodeB = get<1>(edge);
    double weight = get<2>(edge);

    this->addEdge(nodeA,nodeB,weight);
  }
}

///Construct from File - When calling need to cast to string ie Graph G(string("file.txt"));
Graph::Graph(string inputFileName) {
  //Open .txt file
  ifstream file (inputFileName);
  char specialChar = '%';
  char separator = '^';
  string line;

  //If the file is invalid, stop.
  if (!file.is_open()) { return; }

  //Read Header
  getline (file, line);
  if (line == specialChar + string("PERSISTANT GRAPH: DIRECTED (Do not edit this line)")) { directed = true; }
  else if (line == specialChar + string("PERSISTANT GRAPH: UNDIRECTED (Do not edit this line)")) { directed = false; }
  else { return; } //Corrupt File

  getline (file, line);
  if (line != "---------------------------------------------------------") { return; } //Corrupt File

  //Read Node Header
  getline (file, line);
  if (line != specialChar + string("NODES (Do not edit this line):")) { return; } //Corrupt File

  //Read Nodes
  getline (file, line);
  while (line[0] != specialChar) {
    //Split up Node name and Node data using the separator character
    string nodeName = line.substr(0, line.find(separator));
    string dataString = line.substr(line.find(separator)+1);
    double nodeData = stod(dataString);

    //Add Node to Graph, read next line
    addNode(nodeData, nodeName);
    getline (file, line);
  }

  //Read Edges
  if (line != specialChar + string("EDGES (Do not edit this line):")) { return; } //Corrupt File
  while (getline (file, line)) {
    //Split up Edge into sourceNode, targetNode, and weight
    string sourceNode = line.substr(0, line.find(separator));
    line = line.substr(line.find(separator)+1);
    string targetNode = line.substr(0, line.find(separator));
    string weightString = line.substr(line.find(separator)+1);
    double weight = stod(weightString);

    cout << sourceNode << " " << targetNode << " " << weight << endl;

    //Add Edge to Graph
    addEdge(sourceNode, targetNode, weight);
  }
}

Graph::~Graph() {
  for (auto iter : nodeMap) { delete iter.second; }
}

bool Graph::addNode(double data, string name) {
  //If node already exists, return false
  if (nodeMap.find(name) != nodeMap.end()) { return false; }

  //Else, Dynamically Allocate a new Node and put it in 'nodeMap'
  Node* newNode = new Node(data, name);
  nodeMap.emplace(name, newNode);

  return true;
}

bool Graph::addNode(string name) {
  return addNode(1, name);
}

///Given a vector of strings, insert each string as a Node
void Graph::addNodes(vector<string> nodes) {
  for (auto node : nodes) {
    addNode(node);
  }
}

///Given a vector of (double, string) pairs, insert each pair as a Node
void Graph::addNodes(vector<pair<double, string>> nodes) {
  for (auto nodePair : nodes) {
    addNode(nodePair.first, nodePair.second);
  }
}

bool Graph::addEdge(string fromNode, string toNode, double weight) {
  //If one of the nodes don't exist, return false
  if (nodeMap.find(fromNode) == nodeMap.end()) { return false; }
  if (nodeMap.find(toNode) == nodeMap.end()) { return false; }

  //Else add neighbor
  nodeMap[fromNode]->addNeighbor(toNode, weight);
  nodeMap[toNode]->getSetRef().insert(fromNode);

  //If the Graph is undirected, also add the "Inverse-Edge"
  if (!directed) {
    nodeMap[toNode]->addNeighbor(fromNode, weight);
    nodeMap[fromNode]->getSetRef().insert(toNode);
  }

  return true;
}

///Default edge weight is 1
bool Graph::addEdge(string fromNode, string toNode) {
  return addEdge(fromNode, toNode, 1);
}

///Add Edge using a 3-tuple (nodeA,nodeB,weight)
bool Graph::addEdge(tuple<string, string, double> edge) {
  return addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
}

bool Graph::deleteNode(string targetNode) {
  //If node does not exist, return false
  if (nodeMap.find(targetNode) == nodeMap.end()) { return false; }

  //For each Node N in getSetRef(), remove targetNode from N's getMapPtr()
  //getSetRef() will have all Nodes that have an edge to targetNode
  unordered_set<string>& setReference = (nodeMap[targetNode]->getSetRef());
  for (auto iter : setReference) {
    (nodeMap[iter]->getMapPtr())->erase(targetNode);
  }

  //Remove targetNode from it's neighbors "getSetRef()"
  for (auto iter : *(nodeMap[targetNode]->getMapPtr())) {
    nodeMap[iter.first]->getSetRef().erase(targetNode);
  }

  //Deallocate Node, remove it from nodeMap
  delete nodeMap[targetNode];
  nodeMap.erase (targetNode);
  return true;
}

bool Graph::deleteEdge(string fromNode, string toNode, double weight) {
  //If one of the nodes don't exist or no such edge exists, return false
  if (nodeMap.find(fromNode) == nodeMap.end()) { return false; }
  if (nodeMap.find(toNode) == nodeMap.end()) { return false; }
  unordered_map<string, multiset<double>>& neighborMapRef = *(nodeMap[fromNode]->getMapPtr());
  if (neighborMapRef.find(toNode) == neighborMapRef.end()) { return false; }

  //Delete weight from multiset
  multiset<double>& set = neighborMapRef[toNode];
  set.erase(weight);

  //If that was the last edge from fromNode to toNode, delete that (key,value) pair from getMapPtr()
  if (set.empty()) {
    neighborMapRef.erase(toNode);
  }

  //If the Graph is undirected, also delete the "Inverse-Edge"
  if (!directed) {
	  unordered_map<string, multiset<double>>& neighborMapRef1 = *(nodeMap[toNode]->getMapPtr());

	  //Delete weight from multiset
	  multiset<double>& set1 = neighborMapRef1[fromNode];
	  set1.erase(weight);

	  //If that was the last edge from fromNode to toNode, delete that (key,value) pair from getMapPtr()
	  if (set1.empty()) { neighborMapRef1.erase(fromNode);}
  }

  return true;
}

bool Graph::deleteEdge(string fromNode, string toNode) {
  return deleteEdge(fromNode, toNode, 1);
}

///connected: Returns true if the Graph is connected, for undirected Graphs.
bool Graph::connected() {
  if (nodeMap.empty()) { return true;} //An empty Graph is trivially connected

  //Run explore on a random Node
  auto it =  nodeMap.begin();
  unordered_set<string> tempSet = explore(it->first);
  //Is the set of Nodes reachable == # of all Nodes in the Graph?
  return (tempSet.size() == nodeMap.size());

}

///weaklyConnected: Returns true if the Graph is weakly-connected, for directed Graphs.
//A directed graph is called weakly connected if replacing all of its
//directed edges with undirected edges produces a connected (undirected) graph.
bool Graph::weaklyConnected() const {
  if (nodeMap.empty()) { return true;} //An empty Graph is trivially connected

  //Create a copy of this graph
  Graph modifiedCopy(*this);
  //Replace all directed edges with undirected edges (ie for all edges <A,B,w> add <B,A,w>)
  vector< tuple<string, string, double> > edgeVec = modifiedCopy.getEdges();
  for (auto edge : edgeVec) {
    string nodeA = get<0>(edge);
    string nodeB = get<1>(edge);
    double weight = get<2>(edge);
    modifiedCopy.addEdge(nodeB, nodeA, weight);
  }

  //Test if the modified copy is connected
  return modifiedCopy.connected();
}

///stronglyConnected: Returns true if the Graph is strongly-connected, for directed Graphs.
//A directed graph is called strongly connected if
//there is a path in each direction between each pair of vertices of the graph.
bool Graph::stronglyConnected() {
  //DFS on arbitraryNode. If arbitraryNode can't reach all other Nodes, return false.
  string arbitraryNode = nodeMap.begin()->first;
  unordered_set<string> tempSet = explore(arbitraryNode);
  if (tempSet.size() != nodeMap.size()) { return false; }
  //DFS on same arbitraryNode on the transpose of the Graph. If it can reach all other Nodes, the Graph is stronglyConnected.
  Graph T = transpose();
  unordered_set<string> tempSet1 = T.explore(arbitraryNode);
  cout << "***" << tempSet1.size() << endl;
  return (tempSet1.size() == nodeMap.size());
}

///transpose: Returns a Graph object with reversed edges of the original Graph.
Graph Graph::transpose() const {
  //Create a new Graph object.
  Graph graph(directed);

  //Add all existing nodes to the new Graph
  for (auto iter : nodeMap) {
    double data = iter.second->getData();
    graph.addNode(data, iter.first);
  }

  //For all edges A,B,w in the original, add B,A,W to the copy
  vector< tuple<string, string, double> > edgeVec = this->getEdges();
  for (auto edge : edgeVec) {
    string nodeA = get<0>(edge);
    string nodeB = get<1>(edge);
    double weight = get<2>(edge);

    graph.addEdge(nodeB, nodeA, weight);
  }

  return graph;
}


///neighborNames: Returns a list of the names of neighbors
vector<string> Graph::neighborNames(string sourceNode) {
  vector<string> returnVec;

  unordered_map<string, multiset<double>>* neighborMapPtr = nodeMap[sourceNode]->getMapPtr();
  for (auto it : *neighborMapPtr) {
    returnVec.push_back(it.first);
  }

  return returnVec;
}

///neighborDistMin: Returns a list of the names of neighbors along with the lowest edge weight to each neighbor
vector<pair<string, double>> Graph::neighborDistMin(string sourceNode) {
  vector<pair<string, double>> returnVec;

  unordered_map<string, multiset<double>>* neighborMapPtr = nodeMap[sourceNode]->getMapPtr();
  for (auto it : *neighborMapPtr) {
    pair<string, double> tempPair(it.first, *min_element(it.second.begin(),it.second.end()));
    returnVec.push_back(tempPair);
  }

  return returnVec;
}

///neighborDistMax: Returns a list of the names of neighbors along with the highest edge weight to each neighbor
vector<pair<string, double>> Graph::neighborDistMax(string sourceNode) {
  vector<pair<string, double>> returnVec;

  unordered_map<string, multiset<double>>* neighborMapPtr = nodeMap[sourceNode]->getMapPtr();
  for (auto it : *neighborMapPtr) {
    pair<string, double> tempPair(it.first, *max_element(it.second.begin(),it.second.end()));
    returnVec.push_back(tempPair);
  }

  return returnVec;
}

///deleteNeighbors: Removes all neighbors of sourceNode along with all the edges associated with the neighbors.
bool Graph::deleteNeighbors(string sourceNode) {
  if (nodeMap.find(sourceNode) == nodeMap.end()) { return false; }

  vector<string> neighbors = neighborNames(sourceNode);
  for (auto neighbor : neighbors) {
    deleteNode(neighbor);
  }
  return true;
}

///explore: Returns a set of Nodes reachable from sourceNode
unordered_set<string> Graph::explore(string sourceNode) {
  unordered_set<string> reachable; //Will contain all nodes reachable from the passed Node
  exploreHelper(reachable, sourceNode);
  return reachable;
}

void Graph::exploreHelper(unordered_set<string> &visited, string v) {
  visited.insert(v);
  vector<string> neighbors = neighborNames(v);

  for (auto neighbor : neighbors) {
    if (visited.find(neighbor) == visited.end())
      exploreHelper(visited, neighbor);
  }
}

///reachableNames: Returns a list of Nodes reachable from a given sourceNode
vector<string> Graph::reachableNames(string sourceNode) {
  vector<string> returnVec;
  unordered_set<string> reachable = explore(sourceNode);
  for (string name : reachable) {
    returnVec.push_back(name);
  }
  return returnVec;
}

///reachableDists: Returns a list of Nodes and their distances from a given sourceNode (uses BFS)
vector<pair<string, double>> Graph::reachableDists(string sourceNode) {
  double infinity = numeric_limits<double>::max(); //Simulated infinity
  unordered_map<string, double> dist; //Holds the shortest distance to each Node from sourceNode
  vector<pair<string, double>> returnVec;

  //If sourceNode does not exist, return an empty vector
  if (nodeMap.find(sourceNode) == nodeMap.end()) { return returnVec; }

  //For all Nodes N, set dist[N] to infinity
  for (auto iter : nodeMap) {
    dist.emplace(iter.first, infinity);
  }

  //BFS
  dist[sourceNode] = 0;
  queue<string> Q;
  Q.push(sourceNode);

  while (!Q.empty()) {
    string currNode = Q.front();
    Q.pop();
    returnVec.push_back(make_pair (currNode, dist[currNode]));
    //For all Neighbors N of currNode
    vector<string> neighborsCurr = neighborNames(currNode);
    for (auto N : neighborsCurr) {
      if (dist[N] == infinity) {
        Q.push(N);
        dist[N] = dist[currNode] + 1;
      }
    }
  }

  return returnVec;
}

///pathCheck: Returns true if there is a (directed) path from fromNode to toNode.
bool Graph::pathCheck(string sourceNode, string targetNode) {
  unordered_set<string> reachable = explore(sourceNode);
  return (reachable.find(targetNode) != reachable.end());
}

///BFS: Returns the shortest unweighted path from sourceNode to targetNode
vector<string> Graph::BFS(string sourceNode, string targetNode) {
  //If either Node DNE, return an empty vector
  vector<string> pathVec;
  if (nodeMap.find(sourceNode) == nodeMap.end()) { return pathVec; }
  if (nodeMap.find(targetNode) == nodeMap.end()) { return pathVec; }

  //prevMap[X] will contain the Node previous to X. Also keeps track of which Nodes have been visited.
  unordered_map<string, string> prevMap;
  prevMap.emplace(sourceNode, "");

  //BFS
  queue<string> Q;
  Q.push(sourceNode);

  while (!Q.empty()) {
    string currNode = Q.front();
    Q.pop();
    //For all Neighbors N of currNode
    vector<string> neighborsCurr = neighborNames(currNode);
    for (string N : neighborsCurr) {
      if (prevMap.find(N) == prevMap.end()) {
        Q.push(N);
        prevMap.emplace(N, currNode);
      }
    }
  }

  //If the targetNode was not found return an empty vector
  if (prevMap.find(targetNode) == prevMap.end()) { return pathVec; }

  //Use prevMap to get the path from Target back to Source
  string curr = targetNode;
  pathVec.push_back(curr);
  while (true) {
    curr = prevMap[curr];
    if (curr == "") { break; }
    pathVec.push_back(curr);
  }

  //Reverse pathVec so the Node's are in order from Source to Target
  reverse(pathVec.begin(), pathVec.end());

  return pathVec;
}

///DFS: Returns the shortest unweighted path from sourceNode to targetNode
vector<string> Graph::DFS(string sourceNode, string targetNode) {
  //If either Node DNE, return an empty vector
  vector<string> pathVec;
  if (nodeMap.find(sourceNode) == nodeMap.end()) { return pathVec; }
  if (nodeMap.find(targetNode) == nodeMap.end()) { return pathVec; }

  //prevMap[X] will contain the Node previous to X. Also keeps track of which Nodes have been visited.
  unordered_map<string, string> prevMap;
  prevMap.emplace(sourceNode, "");

  //Recursive Kick-Off
  DFShelper(sourceNode, targetNode, prevMap);

  //If the targetNode was not found return an empty vector
  if (prevMap.find(targetNode) == prevMap.end()) { return pathVec; }

  //Use prevMap to get the path from Target back to Source
  string curr = targetNode;
  pathVec.push_back(curr);
  while (true) {
    curr = prevMap[curr];
    if (curr == "") { break; }
    pathVec.push_back(curr);
  }

  //Reverse pathVec so the Node's are in order from Source to Target
  reverse(pathVec.begin(), pathVec.end());

  return pathVec;
}

///DFS - Recursive Function, modifies prevMap
void Graph::DFShelper(string currentNode, string targetNode, unordered_map<string, string> &prevMap) {
  if (currentNode == targetNode) { return; }

  vector<string> neighbors = neighborNames(currentNode);
  for (string neighbor : neighbors) {
    //If this neighbor has not been visited, add it to the prevMap and recurse on it
    if (prevMap.find(neighbor) == prevMap.end()) {
      prevMap.emplace(neighbor, currentNode);
      DFShelper(neighbor, targetNode, prevMap);
    }
  }
}

///Dijktras: Returns the shorted weighted path from sourceNode to targetNode
vector<string> Graph::Dijktras(string sourceNode, string targetNode) {
  double infinity = numeric_limits<double>::max(); //Simulated infinity
  unordered_map<string, double> dist; //Holds the shortest distance to each Node from targetNode
  unordered_map<string, string> prevMap; //Holds the previous node of current node from the source
  vector<string> pathVec;

  if (nodeMap.find(sourceNode) == nodeMap.end()) { return pathVec; }

  //For all Nodes N, set their distance from source to infinity, all prevs are null
  for (auto iter : nodeMap) {
    dist[iter.first] = infinity;
    prevMap[iter.first] = ""; //Empty string serves as null
  }
  dist[sourceNode] = 0;

  //Min-Heap of Pairs, where .first is the shortest distance from source and .second is the name
  //C++ will use the first value of pair as the comparison
  priority_queue<pair<double, string>,
  vector<pair<double, string>>,
  greater<pair<double, string>> > minHeap;

  for (auto iter : nodeMap) {
    minHeap.push(make_pair(dist[iter.first], iter.first));
  }

  //while pQ not empty
  while (!minHeap.empty()) {
    string currNode = minHeap.top().second;
    minHeap.pop();

    //for all neighbors N of currNode
    vector<string> neighborsCurr = neighborNames(currNode);
    for (string N : neighborsCurr) {
      unordered_map<string, multiset<double>>* neighborMapPtr = nodeMap[currNode]->getMapPtr();
      double distanceToN = dist[currNode] + *((*neighborMapPtr)[N]).begin();
      if (dist[N] > distanceToN) {
        dist[N] = distanceToN;
        prevMap[N] = currNode;
      }
    }
  }

  //Use prevMap to get the path from Target back to Source
  string curr = targetNode;
  pathVec.push_back(curr);
  while (true) {
    curr = prevMap[curr];
    if (curr == "") { break; }
    pathVec.push_back(curr);
  }

  //Reverse pathVec so the Node's are in order from Source to Target
  reverse(pathVec.begin(), pathVec.end());

  return pathVec;
}

///Djiktras: Returns an unordered_map where keys are Node names and values are the shortest weighted distance to that Node from sourceNode
unordered_map<string, double> Graph::Dijktras(string sourceNode) {
  double infinity = numeric_limits<double>::max(); //Simulated infinity
  unordered_map<string, double> dist; //Holds the shortest distance to each Node from targetNode
  unordered_map<string, string> prev; //Holds the previous node of current node from the source
  unordered_map<string, double> returnMap; //Holds the distance to all nodes reachable from sourceNode

  if (nodeMap.find(sourceNode) == nodeMap.end()) { return returnMap; }

  //For all Nodes N, set their distance from source to infinity, all prevs are null
  for (auto iter : nodeMap) {
    dist[iter.first] = infinity;
    prev[iter.first] = ""; //Empty string serves as null
  }
  dist[sourceNode] = 0;

  //Min-Heap of Pairs, where .first is the shortest distance from source and .second is the name
  //C++ will use the first value of pair as the comparison
  priority_queue<pair<double, string>,
  vector<pair<double, string>>,
  greater<pair<double, string>> > minHeap;

  for (auto iter : nodeMap) {
    minHeap.push(make_pair(dist[iter.first], iter.first));
  }

  //while pQ not empty
  while (!minHeap.empty()) {
    string currNode = minHeap.top().second;
    minHeap.pop();

    //for all neighbors N of currNode
    vector<string> neighborsCurr = neighborNames(currNode);
    for (string N : neighborsCurr) {
      unordered_map<string, multiset<double>>* neighborMapPtr = nodeMap[currNode]->getMapPtr();
      double distanceToN = dist[currNode] + *((*neighborMapPtr)[N]).begin();
      if (dist[N] > distanceToN) {
        dist[N] = distanceToN;
        prev[N] = currNode;
      }
    }
  }

  for (auto iter : dist) {
    if (iter.second != infinity)
      returnMap.emplace(iter.first, iter.second);
  }
  return returnMap;
}


///BellmanFord: Returns a map where keys are Node names and values are the shortest distance from sourceNode
tuple<unordered_map<string, double>, unordered_map<string, string>, bool> Graph::BellmanFord(string sourceNode) {
  double infinity = numeric_limits<double>::max(); //Simulated infinity
  vector< tuple<string, string, double> > Edges = getEdges();
  bool negativeCycle = false;

  //Initialize Dist & Prev maps
  unordered_map<string, double> Dist; //Holds the shortest distance to each Node from sourceNode
  unordered_map<string, string> Prev; //Holds the previous Node
  for (auto iter : nodeMap) {
    Dist.emplace(iter.first, infinity);
    Prev.emplace(iter.first, "");
  }
  Dist[sourceNode] = 0;

  //Repeatedly "Relax" Edges
  for (int i=1; i <= numNodes()-1; i++) {
    for (auto edge : Edges) {
      string nodeA = get<0>(edge);
      string nodeB = get<1>(edge);
      double weight = get<2>(edge);
      if (Dist[nodeA] == infinity) { continue; } //infinity + weight will overflow so this guards against that
      if (Dist[nodeA] + weight < Dist[nodeB]) {
        Dist[nodeB] = Dist[nodeA] + weight;
        Prev[nodeB] = nodeA;
      }
    }
  }

  //Check for Negative Cycles
  for (auto edge : Edges) {
    string nodeA = get<0>(edge);
    string nodeB = get<1>(edge);
    double weight = get<2>(edge);
    if (Dist[nodeA] == infinity) { continue; } //infinity + weight will overflow so this guards against that
    if (Dist[nodeA] + weight < Dist[nodeB]) {
      //Negative Cycle Detected:
      Prev[nodeA] = nodeB;
      negativeCycle = true;
    }
  }

  //Return
  return make_tuple(Dist, Prev, negativeCycle);
}

unordered_map<string, double> Graph::BellmanFordDist(string sourceNode) {
  return get<0>(BellmanFord(sourceNode));
}
unordered_map<string, string> Graph::BellmanFordPrev(string sourceNode) {
  return get<1>(BellmanFord(sourceNode));
}
bool Graph::NegativeCycle() {
  //Warning! Very inefficient, runs BellmanFord using every Node as a source until a negCycle is detected or none at all.
  for (auto iter : nodeMap) {
    if (get<2>(BellmanFord(iter.first))) {
      return true;
    }
  }
  return false;

}


///Prims: Returns a MST (as a Graph object)
Graph Graph::Prims() {
  //Initialize a tree with a single vertex, chosen arbitrarily from the graph.
  Graph MST;
  if (!connected()) { return MST; } //If the Graph is not connected, return an empty tree.
  string arbitraryNode = nodeMap.begin()->first;
  MST.addNode(arbitraryNode);

  //Repeatedly add the lightest edge until all Nodes are in the tree.
  vector< tuple<string, string, double> > edges = getEdgesAscending();

  while (MST.numEdges() != (numNodes()-1)) { //There are |N-1| Edges in a MST
    for (auto edge : edges) {
      //If one Node is in the tree and the other is not
      if ( (MST.nodeExists(get<0>(edge)) && !MST.nodeExists(get<1>(edge))) ||
           (!MST.nodeExists(get<0>(edge)) && MST.nodeExists(get<1>(edge))) )
      {
        //add Nodes and Edge to MST
        MST.addNode(get<0>(edge));
        MST.addNode(get<1>(edge));
        MST.addEdge(get<0>(edge), get<1>(edge), get<2>(edge));
        break;
      }
    }
  }
  return MST;
}

Graph Graph::Kruskals() {
  //create a graph F (a set of trees), where each vertex in the graph is a separate tree
  Graph MST;
  if (!connected()) { return MST; } //If the Graph is not connected, return an empty tree.

  //Add all nodes in original to new Graph
  for (auto iter : nodeMap) {
    double data = iter.second->getData();
    string name = iter.first;
    MST.addNode(data, name);
  }

  //create a set S containing all the edges in the graph
  vector< tuple<string, string, double> > edges = getEdgesDescending();

  //while S is nonempty and F is not yet spanning
  while (!edges.empty() && MST.numEdges() != (numNodes()-1)) {
    //remove an edge with minimum weight from S
    auto edge = edges.back();
    edges.pop_back();
    string nodeA = get<0>(edge);
    string nodeB = get<1>(edge);
    //if the removed edge connects two different trees then add it to the forest F, combining two trees into a single tree
    if (!MST.pathCheck(nodeA,nodeB)) {
      MST.addNode(nodeA);
      MST.addNode(nodeB);
      MST.addEdge(nodeA,nodeB,get<2>(edge));
    }
  }
  return MST;
}


///getInfo: Returns a string of all Nodes along with their Edges.
string Graph::getInfo() {
  stringstream ss;
  ss << fixed; //Prevents scientific-notation
  ss << "\n\nGraph Info: " << endl;
  //For Every Node
  for (auto iterA : nodeMap) {
    ss << "[" << iterA.first << ": " << iterA.second->getData() << "] ";
    //For Every Neighbor of Node
    for (auto iterB : *(iterA.second->getMapPtr())) {
      ss << "("<< iterB.first << "): ";
      //Print Each Edge of Neighbor
      for (auto weight : iterB.second) {
        ss << weight << ", ";
      }
    }
    ss << "\n\n";
  }
  return ss.str();
}

///getEdges: Returns an unsorted vector of edges, where edges are represented with 3-tuples (nodeA, nodeB, weight)
vector< tuple<string, string, double> > Graph::getEdges() const {
  vector< tuple<string, string, double> > edgeVec;

  //For all Nodes K in nodeMap
  for (auto iter : nodeMap) {
    auto K = iter.second; //K is a Node*
    //For all neighbors N of K
    for (auto iter1 : *(K->getMapPtr())) {
      auto tempSet = iter1.second; //tempSet is an multiset
      //For all weights from K to N, add it to the edgeVec
      for (double i : tempSet) {
        string nodeA = iter.first;
        string nodeB = iter1.first;
        tuple<string, string, double> tempTuple(nodeA, nodeB, i);
        edgeVec.push_back(tempTuple);

      }
    }
  }

  //If the Graph is Undirected, post-process to delete duplicates ie (nodeA,nodeB,w) and (nodeB, nodeA,w)
  if (!directed) {
    //For every (A,B,w) in edgeVec, delete one (B,A,w)
    vector< tuple<string, string, double> > deleteTheseEdges;
    for (auto edge : edgeVec) {
      string nodeA = get<0>(edge);
      string nodeB = get<1>(edge);
      double weight = get<2>(edge);
      tuple<string, string, double> deleteMe(nodeB, nodeA, weight);
      if (nodeA > nodeB) //Prevents deleting both duplicates, we just want to delete one to leave a unique edge.
        deleteTheseEdges.push_back(deleteMe);
    }

    for (auto edge : deleteTheseEdges) {
      edgeVec.erase(remove(edgeVec.begin(), edgeVec.end(), edge), edgeVec.end());
    }
  }


  return edgeVec;
}

///getEdgesAscending: Returns a sorted list of edges from low to high weights
vector< tuple<string, string, double> > Graph::getEdgesAscending() const {
  vector< tuple<string, string, double> > edges = getEdges();

  sort(edges.begin(),edges.end(),
       [](const tuple<string, string, double> & a, const tuple<string, string, double> & b) -> bool
       { return get<2>(a) < get<2>(b); });

  return edges;
}

///getEdgesDescending: Returns a sorted list of edges from high to low weights
vector< tuple<string, string, double> > Graph::getEdgesDescending() const {
  vector< tuple<string, string, double> > edges = getEdges();

  sort(edges.begin(),edges.end(),
       [](const tuple<string, string, double> & a, const tuple<string, string, double> & b) -> bool
       { return get<2>(a) > get<2>(b); });

  return edges;
}

int Graph::numNodes() {
  return nodeMap.size();
}

int Graph::numEdges() {
  return getEdges().size();
}
bool Graph::nodeExists(string name) {
  return (nodeMap.find(name) != nodeMap.end());
}

///saveGraph: Saves a Graph object as a .txt file for later retrieval
void Graph::saveGraph(string outputFileName) { // Compiler was complaining about no return, so changed type to void
  //Prep .txt file
  ofstream output;
  char specialChar = '%';
  char separator = '^';
  output.open (outputFileName+".txt");
  output << fixed; //Prevents scientific-notation

  //Write Header, includes directed bool
  if (directed) { output << specialChar << "PERSISTANT GRAPH: DIRECTED (Do not edit this line)" << endl; }
  else { output << specialChar << "PERSISTANT GRAPH: UNDIRECTED (Do not edit this line)" << endl; }
  output << "---------------------------------------------------------" << endl;

  //Write Nodes
  output << specialChar << "NODES (Do not edit this line):" << endl;
  for (auto iter : nodeMap) {
      output << iter.first << separator << nodeMap[iter.first]->getData() << endl;
  }

  //Write Edges
  output << specialChar << "EDGES (Do not edit this line):" << endl;
  for (auto tuple : getEdges()) {
    output << get<0>(tuple) << separator << get<1>(tuple) << separator << get<2>(tuple) << endl;
  }

  //Close .txt  file
  output.close();
}