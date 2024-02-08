#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Patients {
private:
    string name;
    string bloodType;
    long int phoneNumber;

public:

    Patient(string n, string t, long int num) {
        name = n;
        bloodType = t;
        phoneNumber = num;
    }

    string getName() const {
        return name;
    }

    string getBloodType() const {
        return bloodType;
    }

    long int getPhoneNumber() const {
        return phoneNumber;
    }

};

vector<Patients> getPatients() {
    vector<Patients> patients = {
        Patients("Bill", "AB", 9190136148),  // 1
        Patients("Steve", "B-", 912387781),  // 2
        Patients("Bob", "A+", 6803512987),   // 3
        Patients("Frank", "O", 9192155613)   // 4
    };

    return patients;
}

int main() {
    vector<Patients> patientList = getPatients();

    cout << "Printing Patients list of " << patientList.size() << " patients" << endl;
    for (const Patients& patient : patientList) {
        cout << "Name: " << patient.getName() << ", Blood Type: " << patient.getBloodType() << ", Phone Number: " << patient.getPhoneNumber() << endl;
    }

    return 0;
}
