class MyVector:
    def __init__(self, size=0):
        self.my_array = [None] * size  # Initialize with None values
        self.capacity = size
        self.num_elements = 0

    def push_back(self, ele):
        if self.num_elements == self.capacity:
            self.capacity = 1 if self.capacity == 0 else self.capacity * 2
            new_array = [None] * self.capacity
            for i in range(self.num_elements):
                new_array[i] = self.my_array[i]
            self.my_array = new_array
        self.my_array[self.num_elements] = ele
        self.num_elements += 1

    def insert(self, i, ele):
        if i < 0 or i > self.num_elements:
            print("Invalid index for insertion.")
            return
        if self.num_elements == self.capacity:
            self.capacity = 1 if self.capacity == 0 else self.capacity * 2
            new_array = [None] * self.capacity
            for j in range(i):
                new_array[j] = self.my_array[j]
            new_array[i] = ele
            for j in range(i, self.num_elements):
                new_array[j + 1] = self.my_array[j]
            self.my_array = new_array
        else:
            for j in range(self.num_elements, i, -1):
                self.my_array[j] = self.my_array[j - 1]
            self.my_array[i] = ele
        self.num_elements += 1

    def at(self, i):
        if i < 0 or i >= self.num_elements:
            print("Index out of bounds.")
            return None
        return self.my_array[i]

    def __getitem__(self, i):
        return self.at(i)

    def erase(self, i):
        if i < 0 or i >= self.num_elements:
            print("Index out of bounds.")
            return
        for j in range(i, self.num_elements - 1):
            self.my_array[j] = self.my_array[j + 1]
        self.num_elements -= 1

    def size(self):
        return self.num_elements

    def empty(self):
        return self.num_elements == 0

# Testing the MyVector implementation in Python
if __name__ == "__main__":
    v = MyVector()

    v.push_back(10)
    v.push_back(20)
    print(f"v: {v[0]}, {v.at(1)}")

    v.push_back(30)
    v.push_back(40)
    v.push_back(50)
    v.push_back(60)
    print("v:", [v[i] for i in range(v.size())])
    print(f"Size: {v.size()}, Empty: {v.empty()}")

    v.insert(1, 15)
    v.erase(2)
    print("v:", [v[i] for i in range(v.size())])
    print(f"Size: {v.size()}, Empty: {v.empty()}")

    v2 = MyVector()
    for i in range(v.size()):
        v2.push_back(v[i])
    print("v2:", [v2[i] for i in range(v2.size())])
    print(f"Size: {v2.size()}, Empty: {v2.empty()}")

    v2.erase(0)
    print("v2:", [v2[i] for i in range(v2.size())])
    print(f"Size: {v2.size()}, Empty: {v2.empty()}")

    v3 = MyVector()
    print("v3:", f"Size: {v3.size()}, Empty: {v3.empty()}")
