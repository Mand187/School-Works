import math

class EmployeeData:
    def __init__ (self, name, title, wages, hours):
        self.name = name
        self.title = title
        self.wages = wages
        self.hours = hours
    
    def setName(self, n):
        self.name = n

    def setTitle(self, t):
        self.title = t

    def setWage(self, w):
        if w >= 0.0: 
            self.wages = w
        else: 
            print("Invalid Wage")
            self.wages = 0.0
    
    def setHours(self, h):
        if h >= 0.0:
            self.hours = h
        else:
            print("Invalid hours")
            self.hours = 0.0
    
    def getName(self):
        return self.name
    
    def getTitle(self):
        return self.title
    
    def getWage(self):
        return self.wages
    
    def getHours(self):
        return self.hours
    
    def calculateWages(self):
        return self.wages*self.hours
    
def getEmployee():
    Employees = [
        EmployeeData("Tim Roberts", "Driver", 15.0, 0.0),
        EmployeeData("Matt Jones", "Sales Representative", 15.0, 0.0),
        EmployeeData("Mike Ehrmantraut", "Security Consultant", 25.0, 0.0),
        EmployeeData("Saul Goodman", "Legal Consultant", 35.0, 0.0),
        EmployeeData("Walter White", "Head Chef", 40.0, 0.0)
    ]
    return Employees

def main():
    Employees = getEmployee()
    print("| Employee Database |")
    print("\nGreetings {iUser}, this program displays a list employees and calculates total wages")

    while True:
        print("\n1. Print List of Employees")
        print("2. Calculate Total Wages")
        print("3. Exit Application")

        choice = input("\nMake Selection (1-3): ")

        if choice == '1':
            print("\nList of Employees:")
            for employee in Employees:
                print(f"Name: {employee.getName()}, Title: {employee.getTitle()}, Wage: {employee.getWage()}, Hours: {employee.getHours()}")

        elif choice == '2':
            total_wages = 0.0
            for employee in Employees:
                hours_worked = float(input(f"Enter total hours worked for {employee.getName()}: "))
                employee.setHours(hours_worked)
                total_wages += employee.calculateWages()

            print(f"\nTotal Wages: {total_wages}")

        elif choice == '3':
            print("Exiting the application.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()

