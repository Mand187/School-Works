#include <iostream>
#include <vector>
#include "fooditem.h"

using namespace std;

void displayMenu(const vector<FoodItem>& items) {
    cout << "Menu:\n";
    for (int i = 0; i < items.size(); i++) {
        cout << i + 1 << ": " << items[i].getName() << "\n";
    }
    cout << items.size() + 1 << ": Exit\n";
}

int main() {
    // Pre-populated food items
    vector<FoodItem> items;
    items.push_back(FoodItem("Apple", 52.0, 0.2, 10.3, 0.3, 1.0));
    items.push_back(FoodItem("Banana", 96.0, 0.3, 14.0, 1.1, 1.0));
    items.push_back(FoodItem("Orange", 62.0, 0.2, 12.0, 1.2, 0.0));
    items.push_back(FoodItem("Peach", 59.0, 0.4, 8.4, 1.4, 0.0));
    items.push_back(FoodItem("McNuggets", 250.0, 17.0, 1.5, 14.0, 530.0));
    items.push_back(FoodItem("Instant Ramen", 380.0, 14.0, 1.7, 7.0, 1810.0));
    items.push_back(FoodItem("Pizza", 285.0, 12.0, 3.5, 12.0, 640.0));
    items.push_back(FoodItem("Salad", 120.0, 4.8, 3.5, 6.0, 300.0));
    items.push_back(FoodItem("Hamburger", 250.0, 9.0, 6.0, 15.0, 560.0));
    items.push_back(FoodItem("French Fries", 365.0, 17.0, 0.7, 3.4, 320.0));
    // Add more food items using push_back as needed

    // Total intake for the day
    FoodItem totalIntake;

    // Vector to store selected food items
    vector<FoodItem> selectedItems;

    while (true) {
        displayMenu(items);

        // Read user's input
        int input;
        cin >> input;

        if (input >= 1 && input <= items.size()) {
            selectedItems.push_back(items[input - 1]);
            totalIntake = totalIntake + items[input - 1];
        } else if (input == items.size() + 1) {
            // Exit the loop if "Exit" is selected
            break;
        } else {
            cout << "Invalid input. Please try again.\n";
        }
    }

    // Display the selected food items and their nutritional values
    cout << "\nSelected Food Items:\n";
    for (int i = 0; i < selectedItems.size(); i++) {
        cout << i + 1 << ": " << selectedItems[i].getName() << "\n";
        cout << "   Calories: " << selectedItems[i].getCalories() << " cal\n";
        cout << "   Total Fat: " << selectedItems[i].getTotalFat() << " g\n";
        cout << "   Total Sugars: " << selectedItems[i].getTotalSugars() << " g\n";
        cout << "   Protein: " << selectedItems[i].getProtein() << " g\n";
        cout << "   Sodium: " << selectedItems[i].getSodium() << " mg\n";
    }

    // Display the total nutritional value for the day if items were selected
    if (!selectedItems.empty()) {
        cout << "\nTotal Nutritional Value for the Day:\n";
        cout << "Calories: " << totalIntake.getCalories() << " cal\n";
        cout << "Total Fat: " << totalIntake.getTotalFat() << " g\n";
        cout << "Total Sugars: " << totalIntake.getTotalSugars() << " g\n";
        cout << "Protein: " << totalIntake.getProtein() << " g\n";
        cout << "Sodium: " << totalIntake.getSodium() << " mg\n";

        // Check if the recommended intakes are exceeded
        if (totalIntake.getCalories() > 2000.0) {
            cout << "Exceeded recommended energy intake.\n";
        }
        if (totalIntake.getTotalFat() > 70.0) {
            cout << "Exceeded recommended total fat intake.\n";
        }
        if (totalIntake.getTotalSugars() > 30.0) {
            cout << "Exceeded recommended total sugars intake.\n";
        }
        if (totalIntake.getProtein() > 50.0) {
            cout << "Exceeded recommended protein intake.\n";
        }
        if (totalIntake.getSodium() > 2300.0) {
            cout << "Exceeded recommended sodium intake.\n";
        }
    }

    return 0;
}
