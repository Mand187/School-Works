#include "Graph.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

Graph airportGraph(false);  // Declare the graph globally

void addNode() {
  // Add airports with runway distances in nautical miles
  airportGraph.addNode(2.38,"CLT");   // CLT has a runway distance of 50 nautical miles
  airportGraph.addNode(2.5, "ATL");   // ATL has a runway distance of 55 nautical miles
  airportGraph.addNode(1.6, "DFW");   // DFW has a runway distance of 80 nautical miles
  airportGraph.addNode(2.0, "DEN");   // DEN has a runway distance of 45 nautical miles
  airportGraph.addNode(2.2, "SFO");   // SFO has a runway distance of 65 nautical miles
  airportGraph.addNode(1.7, "MIA");   // MIA has a runway distance of 70 nautical miles
  airportGraph.addNode(1.8, "SEA");   // SEA has a runway distance of 55 nautical miles
  airportGraph.addNode(2.3, "MCO");   // MCO has a runway distance of 50 nautical miles
  airportGraph.addNode(2.7, "PHX");
  airportGraph.addNode(2.1, "BOS");
  airportGraph.addNode(1.9, "LAS");
  airportGraph.addNode(3.0, "LAX");
  
}

void addEdge() {
  // Connect airports with appropriate distances in nautical miles
  airportGraph.addEdge("CLT", "ATL", 150.0);   // Distance between CLT and ATL is 150 nautical miles
  airportGraph.addEdge("CLT", "DFW", 350.0);   // Distance between CLT and DFW is 300 nautical miles
  
  airportGraph.addEdge("DFW", "DEN", 350.0);   // Distance between DFW and DEN is 350 nautical miles
  airportGraph.addEdge("DFW", "SFO", 500.0);   // Distance between DFW and SFO is 500 nautical miles
  airportGraph.addEdge("DFW", "MIA", 250.0);   // Distance between DFW and MIA is 250 nautical miles
  
  airportGraph.addEdge("ATL", "SEA", 200.0);   // Distance between ATL and SEA is 200 nautical miles
  airportGraph.addEdge("ATL", "MCO", 400.0);   // Distance between ATL and MCO is 400 nautical miles
  airportGraph.addEdge("ATL", "PHX", 450.0);   // Distance between ATL and PHX is 450 nautical miles
  
  airportGraph.addEdge("DEN", "SFO", 550.0);   // Distance between DEN and SFO is 550 nautical miles
  airportGraph.addEdge("DEN", "MCO", 300.0);   // Distance between DEN and MCO is 300 nautical miles
  airportGraph.addEdge("DEN", "PHX", 600.0);   // Distance between DEN and PHX is 600 nautical miles
  
  airportGraph.addEdge("SFO", "DEN", 550.0);   // Distance between SFO and DEN is 550 nautical miles
  airportGraph.addEdge("SFO", "MCO", 250.0);   // Distance between SFO and MCO is 250 nautical miles
  airportGraph.addEdge("SFO", "MIA", 700.0);   // Distance between SFO and MIA is 700 nautical miles
  
  airportGraph.addEdge("MCO", "SFO", 250.0);   // Distance between MCO and SFO is 250 nautical miles
  airportGraph.addEdge("MCO", "SEA", 350.0);   // Distance between MCO and SEA is 350 nautical miles
  airportGraph.addEdge("MCO", "BOS", 600.0);   // Distance between MCO and BOS is 600 nautical miles
  
  airportGraph.addEdge("SEA", "BOS", 800.0);   // Distance between SEA and BOS is 800 nautical miles
  airportGraph.addEdge("SEA", "PHX", 550.0);   // Distance between SEA and PHX is 550 nautical miles
  airportGraph.addEdge("SEA", "LAS", 400.0);   // Distance between SEA and LAS is 400 nautical miles
  
  airportGraph.addEdge("BOS", "PHX", 750.0);   // Distance between BOS and PHX is 750 nautical miles
  airportGraph.addEdge("BOS", "LAS", 700.0);   // Distance between BOS and LAS is 700 nautical miles
  airportGraph.addEdge("BOS", "ATL", 900.0);   // Distance between BOS and ATL is 900 nautical miles
}

int main() {
  addNode();
  addEdge();

  // Use Dijkstra's algorithm to find the shortest path from CLT to SEA
  vector<string> shortestPath = airportGraph.Dijktras("CLT", "MCO");

  // Print information about the graph
  cout << airportGraph.getInfo();

  // Print the shortest path
  cout << "Dijktras" << endl;
  for (int i = 0; i < shortestPath.size(); i++) {
    if (i == shortestPath.size() - 1) {
      cout << shortestPath[i];
    } else {
      cout << shortestPath[i] << " -> ";
    }
  }
  
   cout << endl;

vector<string> shortestPathBFS = airportGraph.BFS("CLT","MCO");
  cout << "BFS" << endl;
  for (int i = 0; i < shortestPathBFS.size(); i++) {
    if (i == shortestPathBFS.size() - 1) {
      cout << shortestPathBFS[i];
    } else {
      cout << shortestPathBFS[i] << " -> ";
    }
  }
  
 cout << endl;
  
vector<string> shortestPathDFS = airportGraph.DFS("CLT","MCO");
    cout << "DFS" << endl;
    for (int i = 0; i < shortestPathDFS.size(); i++) {
    if (i == shortestPathDFS.size() - 1) {
      cout << shortestPathDFS[i];
    } else {
      cout << shortestPathDFS[i] << " -> ";
    }
  }

  cout << endl;

  return 0;
}
