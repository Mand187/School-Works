#include "fooditem.h"

FoodItem::FoodItem() {
    name = "";
    calories = 0.0;
    totalFat = 0.0;
    totalSugars = 0.0;
    protein = 0.0;
    sodium = 0.0;
}

FoodItem::FoodItem(std::string name, double calories, double totalFat, double totalSugars, double protein, double sodium) {
    this->name = name;
    this->calories = calories;
    this->totalFat = totalFat;
    this->totalSugars = totalSugars;
    this->protein = protein;
    this->sodium = sodium;
}

// Accessors
std::string FoodItem::getName() const {
    return name;
}

double FoodItem::getCalories() const {
    return calories;
}

double FoodItem::getTotalFat() const {
    return totalFat;
}

double FoodItem::getTotalSugars() const {
    return totalSugars;
}

double FoodItem::getProtein() const {
    return protein;
}

double FoodItem::getSodium() const {
    return sodium;
}

// Mutators
void FoodItem::setName(std::string name) {
    this->name = name;
}

void FoodItem::setCalories(double calories) {
    this->calories = calories;
}

void FoodItem::setTotalFat(double totalFat) {
    this->totalFat = totalFat;
}

void FoodItem::setTotalSugars(double totalSugars) {
    this->totalSugars = totalSugars;
}

void FoodItem::setProtein(double protein) {
    this->protein = protein;
}

void FoodItem::setSodium(double sodium) {
    this->sodium = sodium;
}

// Addition operator overloading
FoodItem FoodItem::operator+(const FoodItem& other) const {
    FoodItem result;
    result.setName(name + " + " + other.name);
    result.setCalories(calories + other.calories);
    result.setTotalFat(totalFat + other.totalFat);
    result.setTotalSugars(totalSugars + other.totalSugars);
    result.setProtein(protein + other.protein);
    result.setSodium(sodium + other.sodium);
    return result;
}
