#include <iostream>
#include <vector>
#include "linked.h"

using namespace std;

LinkedList mergeLists(const LinkedList& list1, const LinkedList& list2) {
    LinkedList mergedList;

    int i = 0; // Index for list1
    int j = 0; // Index for list2

    // Merge while both lists have elements
    while (i < list1.size() && j < list2.size()) {
        int x = list1.at(i);
        int y = list2.at(j);

        if (x <= y) {
            mergedList.push_back(x);
            i++;
        }
        else {
            mergedList.push_back(y);
            j++;
        }
    }

    // If there are remaining elements in list1, append them
    while (i < list1.size()) {
        mergedList.push_back(list1.at(i));
        i++;
    }

    // If there are remaining elements in list2, append them
    while (j < list2.size()) {
        mergedList.push_back(list2.at(j));
        j++;
    }

    cout << endl << "Merged List" << endl << "[";
    for (int k = 0; k < mergedList.size(); k++) {
        cout << mergedList.at(k);
        if (k < mergedList.size() - 1) {
            cout << ",";
        }
    }

    cout << "]" << endl;

    return mergedList;
}




int main() {
    LinkedList myList1; // Create the first instance of LinkedList
    LinkedList myList2; // Create the second instance of LinkedList

    myList1.push_back(2); // Add values to myList1
    myList1.push_back(3);
    myList1.push_back(4);

    cout << endl << "List 1" << endl << "[";
    for (int k = 0; k < myList1.size(); k++) {
        cout << myList1.at(k);
        if (k < myList1.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl;
    
    myList2.push_back(1);
    myList2.push_back(3);
    myList2.push_back(4);

    cout << endl << "List 2" << endl << "[";
    for (int k = 0; k < myList2.size(); k++) {
        cout << myList2.at(k);
        if (k < myList2.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl;

    // Merge myList2 into myList1 using the mergeLists function
    mergeLists(myList1, myList2);

    return 0;
}