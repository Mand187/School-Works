{"filter":false,"title":"menu.cpp","tooltip":"/PRO04/menu.cpp","undoManager":{"mark":2,"position":2,"stack":[[{"start":{"row":0,"column":0},"end":{"row":30,"column":0},"action":"insert","lines":["#include \"menu.h\"","","Menu::Menu() {","    // Populate the menu with pre-defined food items","    foodItems.push_back(FoodItem(\"Apple\", 52, 0, 10, 0, 0));","    foodItems.push_back(FoodItem(\"French Fries\", 365, 17, 0, 3, 246));","    foodItems.push_back(FoodItem(\"Burger\", 250, 10, 6, 13, 380));","    foodItems.push_back(FoodItem(\"Instant Ramen\", 380, 14, 2, 8, 1850));","    foodItems.push_back(FoodItem(\"Protein Bar\", 200, 8, 15, 20, 180));","}","","void Menu::addFoodItem(const FoodItem& foodItem) {","    foodItems.push_back(foodItem);","}","","void Menu::displayMenu() const {","    std::cout << \"Menu:\" << std::endl;","    for (size_t i = 0; i < foodItems.size(); ++i) {","        std::cout << (i + 1) << \": \" << foodItems[i].getName() << std::endl;","    }","    std::cout << (foodItems.size() + 1) << \": Finished\" << std::endl;","}","","bool Menu::isValidOption(int option) const {","    return (option >= 1 && option <= foodItems.size() + 1);","}","","FoodItem Menu::getFoodItem(int option) const {","    return foodItems[option - 1];","}",""],"id":1}],[{"start":{"row":29,"column":1},"end":{"row":30,"column":0},"action":"insert","lines":["",""],"id":2},{"start":{"row":30,"column":0},"end":{"row":31,"column":0},"action":"insert","lines":["",""]}],[{"start":{"row":31,"column":0},"end":{"row":34,"column":0},"action":"insert","lines":["std::vector<FoodItem> Menu::getFoodItems() const {","    return foodItems;","}",""],"id":3}]]},"ace":{"folds":[],"scrolltop":0,"scrollleft":0,"selection":{"start":{"row":34,"column":0},"end":{"row":34,"column":0},"isBackwards":false},"options":{"guessTabSize":true,"useWrapMode":false,"wrapToView":true},"firstLineState":0},"timestamp":1689603329807,"hash":"a16aebc7c806885ca518332d32c46d3394ad2917"}