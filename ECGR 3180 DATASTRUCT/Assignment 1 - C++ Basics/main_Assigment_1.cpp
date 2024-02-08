#include <iostream>

using namespace std;

template <typename T>
class MyVector {
private:
    T* myArray;        // Pointer to dynamic array
    int capacity;      // Max capacity of array
    int numElements;   // Current number of elements

public:
    MyVector() {       // Default constructor
        myArray = nullptr;
        capacity = 0;
        numElements = 0;
    }

    MyVector(int sz) {  // Constructor with initial size
        myArray = new T[sz];
        capacity = sz;
        numElements = 0;
    }

    MyVector(const MyVector& vin) {  // Copy constructor
        capacity = vin.capacity;
        numElements = vin.numElements;
        myArray = new T[capacity];
        for (int i = 0; i < numElements; i++) {
            myArray[i] = vin.myArray[i];
        }
    }

    MyVector& operator=(const MyVector& vin) {  // Copy assignment operator
        if (this != &vin) {
            delete[] myArray;
            capacity = vin.capacity;
            numElements = vin.numElements;
            myArray = new T[capacity];
            for (int i = 0; i < numElements; i++) {
                myArray[i] = vin.myArray[i];
            }
        }
        return *this;
    }

    void pushBack(const T& ele) {  // Add element to end
        if (numElements == capacity) {
            capacity = (capacity == 0) ? 1 : capacity * 2;
            T* newArr = new T[capacity];
            for (int i = 0; i < numElements; i++) {
                newArr[i] = myArray[i];
            }
            delete[] myArray;
            myArray = newArr;
        }
        myArray[numElements++] = ele;
    }

    void insert(int i, const T& ele) {  // Insert element at index
        if (i < 0 || i > numElements) {
            cout << "Invalid index for insertion.\n";
            return;
        }
        if (numElements == capacity) {
            capacity = (capacity == 0) ? 1 : capacity * 2;
            T* newArr = new T[capacity];
            for (int j = 0; j < i; j++) {
                newArr[j] = myArray[j];
            }
            newArr[i] = ele;
            for (int j = i; j < numElements; j++) {
                newArr[j + 1] = myArray[j];
            }
            delete[] myArray;
            myArray = newArr;
        } else {
            for (int j = numElements; j > i; j--) {
                myArray[j] = myArray[j - 1];
            }
            myArray[i] = ele;
        }
        numElements++;
    }

    T at(int i) const {  // Access element at index
        if (i < 0 || i >= numElements) {
            cout << "Index out of bounds.\n";
            return T();
        }
        return myArray[i];
    }

    T operator[](int i) const {  // Array-like access
        return at(i);
    }

    void erase(int i) {  // Erase element at index
        if (i < 0 || i >= numElements) {
            cout << "Index out of bounds.\n";
            return;
        }
        for (int j = i; j < numElements - 1; j++) {
            myArray[j] = myArray[j + 1];
        }
        numElements--;
    }

    int size() const {  // Current size
        return numElements;
    }

    bool empty() const {  // Check if empty
        return numElements == false;
    }

    ~MyVector() {  // Destructor
        delete[] myArray;
    }
};

int main() {
    MyVector<int> v;

    v.pushBack(10);
    v.pushBack(20);
    cout << "v: " << v[0] << ", " << v.at(1) << endl;

    v.pushBack(30);
    v.pushBack(40);
    v.pushBack(50);
    v.pushBack(60);
    cout << "v: ";
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "\nSize: " << v.size() << ", Empty: " << v.empty() << endl;

    v.insert(1, 15);
    v.erase(2);
    cout << "v: ";
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << "\nSize: " << v.size() << ", Empty: " << v.empty() << endl;

    MyVector<int> v2 = v;
    cout << "v2: ";
    for (int i = 0; i < v2.size(); i++) {
        cout << v2[i] << " ";
    }
    cout << "\nSize: " << v2.size() << ", Empty: " << v2.empty() << endl;

    v2.erase(0);
        cout << "v2: ";
    for (int i = 0; i < v2.size(); i++) {
        cout << v2[i] << " ";
    }
    cout << "\nSize: " << v2.size() << ", Empty: " << v2.empty() << endl;

    MyVector<int> v3;
    cout << "v3: ";
    cout << "Size: " << v3.size() << ", Empty: " << v3.empty() << endl;


    return 0;
}
