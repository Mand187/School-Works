from FoodItem import FoodItem

def getFoodItems():
    foodMenu = [
        FoodItem("Apple", 52.0, 0.2, 10.3, 0.3, 1.0),
        FoodItem("Banana", 96.0, 0.3, 14.0, 1.1, 1.0),
        FoodItem("Orange", 62.0, 0.2, 12.0, 1.2, 0.0),
        FoodItem("Peach", 59.0, 0.4, 8.4, 1.4, 0.0),
        FoodItem("McNuggets", 250.0, 17.0, 1.5, 14.0, 530.0),
        FoodItem("Instant Ramen", 380.0, 14.0, 1.7, 7.0, 1810.0),
        FoodItem("Pizza", 285.0, 12.0, 3.5, 12.0, 640.0),
        FoodItem("Salad", 120.0, 4.8, 3.5, 6.0, 300.0),
        FoodItem("Hamburger", 250.0, 9.0, 6.0, 15.0, 560.0),
        FoodItem("French Fries", 365.0, 17.0, 0.7, 3.4, 320.0),
    ]

    return foodMenu

def main():
    foodMenu = getFoodItems()

    for i, food in enumerate(foodMenu, start=1):
        print(f"{i}. Name: {food.getName()}, Calories: {food.getCalories()}, Total Fat: {food.getTotalFat()}, Total Sugars: {food.getTotalSugars()}, Protein: {food.getProtein()}, Sodium: {food.getSodium()}")

    try:
        # Get user input for the first location selection
        first_location_index = int(input("Select the first location (enter the corresponding number): ")) - 1
        firstitem = getFoodItems[first_location_index]

if __name__ == "__main__":
    main()
