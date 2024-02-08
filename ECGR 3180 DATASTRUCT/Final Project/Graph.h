//Min Chang
//Github: Minyc510

#ifndef GRAPH_H
#define GRAPH_H

#include "Node.h"
#include <vector>
#include <utility>
#include <tuple>

using namespace std;

class Graph {

private:
  unordered_map<string, Node*> nodeMap;
  bool directed = true;

public:
  //Constructors & Destructor
  Graph(); //Graphs are directed by default
  Graph(bool directed);
  Graph(const Graph& other); //Copy-Constructor, uses getEdges function
  Graph(string inputFileName); //Constructs Graph from .txt file
  ~Graph();

  //Trivial Functions
  // Node Operations 
  bool addNode(double data, string name); // Creates node with a data value and a name
  bool addNode(string name); //Default data-value '1' (Creates a node with no provided data)
  void addNodes(vector<string> nodes); // Uses vector data to create a group of Nodes, only takes string name 
  void addNodes(vector<pair<double, string>> nodes); // Uses vector data to create a group of nodes takes string name and data in the form of a double
/*
  Explanation of Edges: 
  In simple terms, edges represent a connection between two nodes.

  - Undirected Edges:
    Implies a bidirectional connection. If there is an edge between A and B, 
    it follows that B and A also have an edge connection.

  - Directed Edges:
    The edge connections are one-way, so a connection between A and B does not 
    imply a connection between B and A. The edge connections have to be specified.

  - Weighted Edges:
    Represented by some numerical value indicating cost, distance, or another significance.
    For the program and computer, it is just another variable storing data and
    doesnt HAVE to mean anything.
*/
  // Edge Operations 
  bool addEdge(string fromNode, string toNode, double weight); // Connects from nodes A to B with a given weight
  bool addEdge(string fromNode, string toNode); //Default weight '1' (Connects from nodes A to B with)
  bool addEdge(tuple<string, string, double> edge); // Tuple is datastructure with a fixed size, can contain multiple data types 
  // Delte Nodes 
  bool deleteNode(string targetNode); // Self Explanatory 
  bool deleteEdge(string fromNode, string toNode, double weight);
  bool deleteEdge(string fromNode, string toNode); //Default weight '1'

  //Undirected Graph Specific Functions
  bool connected(); //Is the Graph connected? (A connected graph has a path between every pair of nodes)

  //Directed Graph Specific Functions
  bool weaklyConnected() const; // Graph is connected but doesnt check direction 
  bool stronglyConnected(); //  Strong connectivity requires a directed path between every pair of nodes.

  //Modification Functions
  Graph transpose() const; //Creates a copy, reverses edges of that copy and returns it.
  
  //Neighbor Functions
  // Returns a vector of names of neighboring nodes for the specified node.
  vector<string> neighborNames(string name);
  // Returns a vector of pairs, where each pair represents a neighboring node and its minimum distance to the specified node.
  vector<pair<string, double>> neighborDistMin(string name);
  // Returns a vector of pairs, where each pair represents a neighboring node and its maximum distance to the specified node.
  vector<pair<string, double>> neighborDistMax(string name);
  bool deleteNeighbors(string name); // Deletes all neighbors of the specified node.

  //Explore Functions
  unordered_set<string> explore(string sourceNode); //Returns a set of Nodes reachable from the source Node
  void exploreHelper(unordered_set<string> &visited, string name);
  vector<string> reachableNames(string sourceNode); //Returns a list of Nodes that are reachable from the target
  vector<pair<string, double>> reachableDists(string sourceNode); //Includes distances
  bool pathCheck(string fromNode, string toNode);

/*
  - Breadth-First Search (BFS):
    Traverses the graph level by level.
    Enqueues neighbors, explores the current level, then moves to the next level.

  - Depth-First Search (DFS):
    Traverses the graph by exploring as far as possible before backtracking.
    Aims to discover both long and short paths in the process.

  - Dijkstra's Algorithm:
    Finds the shortest path between two nodes in a weighted graph.
    Explores nodes with the smallest distance, updates distance if a shorter path is found, and repeats until the target is reached.
*/

  //Core Graph Functions
  vector<string> BFS(string sourceNode, string targetNode); //Returns the shortest path from source to target
  vector<string> DFS(string sourceNode, string targetNode); //Returns the shortest path from source to target
  void DFShelper(string sourceNode, string targetNode, unordered_map<string, string> &prevMap);
  vector<string> Dijktras(string sourceNode, string targetNode); //Returns the shortest path from source to target
  unordered_map<string, double> Dijktras(string sourceNode); //Returns a map where keys are nodes reachable from source and values are the shortest distance from source

  //BellmanFord: Returns a 3-tuple containing the Dist and Prev maps, as well as a boolean for the existence of a negative cycle
  tuple<unordered_map<string, double>, unordered_map<string, string>, bool> BellmanFord(string sourceNode);
  unordered_map<string, double> BellmanFordDist(string sourceNode); //Returns just the Dist map
  unordered_map<string, string> BellmanFordPrev(string sourceNode); //Returns just the Prev map
  bool NegativeCycle(); //Does the graph contain a negCycle? Warning!: Exponential Run-Time

  //MST Functions
  Graph Prims();
  Graph Kruskals();

  //About Graph Functions
  string getInfo(); //Returns a list of all Nodes along with their Edges.
  vector< tuple<string, string, double> > getEdges() const; //Returns a vector of Edges, where Edges are represented with a 3-tuple (nodeA,nodeB,weight)
  vector< tuple<string, string, double> > getEdgesAscending() const; // Returns edges in ascending order of weights.
  vector< tuple<string, string, double> > getEdgesDescending() const; // Returns edges in descending order of weights.
  int numNodes(); //Returns the number of Nodes
  int numEdges();
  bool nodeExists(string node); //Is the Node in the Graph?

  //Persistent Graph Functions
  void saveGraph(string outputFileName); // Changed type to void

};

#endif // GRAPH_H