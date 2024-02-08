#include <iostream>
#include <cmath>
#include <string>

using namespace std;

class vectorMag2d {
private:
    string name;
    double xPoint;
    double yPoint;

public:
    // Constructor to initialize 
    vectorMag2d(string n, double x, double y) {
        name = n;
        xPoint = x;
        yPoint = y;
    }

    // Accessor 
    string getName() const {
        return name;
    }

    // Accessor
    double getXPoint() const {
        return xPoint;
    }

    // Accessor 
    double getYPoint() const {
        return yPoint;
    }

    // Mutator 
    void setXPoint(double x) {
        xPoint = x;
    }

    // Mutator 
    void setName(string n) {
        name = n;
    }

    // Mutator 
    void setYPoint(double y) {
        yPoint = y;
    }

    // Public function
    double calcMagnitude() const {
        return sqrt(xPoint * xPoint + yPoint * yPoint);
    }
};

int main() {
int x = 42;
int* p = &x;
cout << *p << endl;
    return 0;
}
