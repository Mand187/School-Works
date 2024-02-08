#ifndef FOODITEM_H
#define FOODITEM_H

#include <iostream>
#include <string>

class FoodItem {
private:
    std::string name;
    double calories;
    double totalFat;
    double totalSugars;
    double protein;
    double sodium;

public:
    FoodItem();
    FoodItem(std::string name, double calories, double totalFat, double totalSugars, double protein, double sodium);

    // Accessors
    std::string getName() const;
    double getCalories() const;
    double getTotalFat() const;
    double getTotalSugars() const;
    double getProtein() const;
    double getSodium() const;

    // Mutators
    void setName(std::string name);
    void setCalories(double calories);
    void setTotalFat(double totalFat);
    void setTotalSugars(double totalSugars);
    void setProtein(double protein);
    void setSodium(double sodium);

    // Addition operator overloading
    FoodItem operator+(const FoodItem& other) const;
};

#endif
