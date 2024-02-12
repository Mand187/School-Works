#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Employee {
private:
    string name;
    string title;
    double wages;
    double hours;

public:
    void setName(string n) {
        name = n;
    }

    string getName() const {
        return name;
    }

    void setTitle(string t) {
        title = t;
    }

    string getTitle() const {
        return title;
    }

    void setWages(double w) {
        if (w >= 0.0) {
            wages = w;
        } else {
            cout << "Invalid wages" << endl;
            wages = 0.0;
        }
    }

    double getWages() const {
        return wages;
    }

    void setHours(double h) {
        if (h >= 0.0) {
            hours = h;
        } else {
            cout << "Invalid hours" << endl;
            hours = 0.0;
        }
    }

    double getHours() const {
        return hours;
    }

    double calculateSalary() const {
        return wages * hours;
    }
};

vector<Employee> getEmployees() {
    vector<Employee> employees;
    
    Employee emp0;
    emp0.setName("Tim Roberts");
    emp0.setTitle("Driver");
    emp0.setWages(15.0);
    emp0.setHours(0.0); // Set initial hours to 0
    employees.push_back(emp0);

    Employee emp1;
    emp1.setName("Matt Jones");
    emp1.setTitle("Sales Representative");
    emp1.setWages(15.0);
    emp1.setHours(0.0); // Set initial hours to 0
    employees.push_back(emp1);

    Employee emp2;
    emp2.setName("Mike Ehrmantraut");
    emp2.setTitle("Security Consultant");
    emp2.setWages(25.0);
    emp2.setHours(0.0); // Set initial hours to 0
    employees.push_back(emp2);

    Employee emp3;
    emp3.setName("Saul Goodman");
    emp3.setTitle("Legal Consultant");
    emp3.setWages(35.0);
    emp3.setHours(0.0); // Set initial hours to 0
    employees.push_back(emp3);

    Employee emp4;
    emp4.setName("Walter White");
    emp4.setTitle("Head Chef");
    emp4.setWages(40.0);
    emp4.setHours(0.0); // Set initial hours to 0
    employees.push_back(emp4);
    
    return employees;
}



int main() {
    int userChoice;
    string iUser = "<User>";
    vector<Employee> dataEmployees = getEmployees();
    double userHours;
    double totalWages = 0.0;

    while (true) {
        cout << "| Employee Database |" << endl;
        cout << "Greetings " << iUser << ", this program displays a list of employees and calculates total wages." << endl;
        cout << "Please choose an option below" << endl;
        cout << "1. Print List of Employees" << endl;
        cout << "2. Calculate Total Wages" << endl;
        cout << "3. Exit Application" << endl << endl;
        cout << "Make Selection: ";
        cin >> userChoice;
        cout << endl;

        if (userChoice == 1) {
            cout << "Printing Employee list of " << dataEmployees.size() << " employees" << endl;
            int vectorNum = 1;
            for (const Employee& employee : dataEmployees) {
                cout << vectorNum++ << ". ";
                cout << "Name: " << employee.getName();
                cout << " Title: " << employee.getTitle();
                cout << " Wage (Hourly): $" << employee.getWages();
                cout << endl;
            }
            cout << endl;
        } else if (userChoice == 2) {
            totalWages = 0.0;
            cout << "Calculating Wages for " << dataEmployees.size() << " employees" << endl;
            for (Employee& employee : dataEmployees) {
                cout << "Enter total hours worked for employee " << employee.getName() << ": ";
                cin >> userHours;
                if (userHours < 0) {
                    cout << "Invalid hours. Please enter a non-negative value." << endl;
                    continue;
                }
                employee.setHours(userHours);
                totalWages += employee.calculateSalary();
                cout << "Employee " << employee.getName() << " earned: $" << employee.calculateSalary() << endl;
            }
            cout << "Total Wages: $" << totalWages << endl;
        } else if (userChoice == 3) {
            cout << "Exiting Program" << endl;
            break;
        } else {
            cout << "Invalid Selection. Please Try Again" << endl;
        }

        cout << endl;
        cout << "1. Return to Menu" << endl;
        cout << "2. Exit Program" << endl;
        cout << "Make Selection: ";
        cin >> userChoice;
        cout << endl;

        if (userChoice == 2) {
            cout << "Exiting Program" << endl;
            break;
        }
    }

    return 0;
}
