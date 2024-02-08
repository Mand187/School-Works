// From Goodrich, Tamassia, and Mount
// Homework

/*
 Write a short C++ program that takes two arguments of type STL vec-
tor<double>, a and b, and returns the element-by-element product of a
and b. That is, it returns a vector c of the same length such that c[i] =
a[i] · b[i].
*/

#include <iostream>
#include <vector>

using namespace std;

// Function for element-wise product of two vectors
vector<double> vecProduct(const vector<double> &v1, const vector<double> &v2) {
    if (v1.size() != v2.size()) {
        return vector<double>();  // Return empty vector if sizes differ
    }

    vector<double> result = {};  // Store result of element-wise product

    for (int i = 0; i < v1.size(); i++) {
        result.push_back(v1.at(i) * v2.at(i));  // Calculate and store product
    }

    return result;  // Return element-wise product vector
}

// Overload << operator to print vectors
ostream &operator<<(ostream &os, const vector<double> &v) {
    for (int i = 0; i < v.size(); i++) {
        os << v.at(i) << " ";  // Print vector elements
    }
    os << endl;
    return os;
}

int main() {
    vector<double> v1{1.0, 2.0, 3.0};
    vector<double> v2{4.0, 5.0, 6.0};
    vector<double> v3 = vecProduct(v1, v2);
    cout << v3 << endl;  // Print element-wise product: 4, 10, 18

    vector<double> v4{42.0};
    cout << vecProduct(v1, v4) << endl;  // Print empty vector
}
