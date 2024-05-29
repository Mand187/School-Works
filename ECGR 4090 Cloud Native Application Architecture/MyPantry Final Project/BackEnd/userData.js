class User {
    constructor(uid, username, firstName, lastName) {
        this.uid = uid;
        this.username = username;
        this.firstName = firstName;
        this.lastName = lastName;
        this.pantryItems = {};
        this.recipes = {};
        this.listItems = {};
    }

    // Method to add pantry items
    addPantryItems(category, items) {
        if (!this.pantryItems[category]) {
            this.pantryItems[category] = [];
        }
        this.pantryItems[category].push(...items);
    }

    // Method to add recipes
    addRecipes(mealType, recipes) {
        if (!this.recipes[mealType]) {
            this.recipes[mealType] = [];
        }
        this.recipes[mealType].push(...recipes);
    }

    // Method to add list items
    addShoppingListItems(category, items) {
        if (!this.listItems[category]) {
            this.listItems[category] = [];
        }
        this.listItems[category].push(...items);
    }

    // Method to fetch pantry items
    fetchPantryItems() {
        return this.pantryItems;
    }

    // Method to fetch recipes
    fetchRecipes() {
        return this.recipes;
    }

    // Method to fetch list items
    fetchShoppingListItems() {
        return this.listItems;
    }
}

const user = new User('101000', 'test', "testUser", "firstTestUser");


// Fetch recipe data
fetch('./spoonRecipe.json')
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(recipeData => {
    // Read the contents of the JSON file
    const title = recipeData.title;
    const ingredients = recipeData.extendedIngredients;
    const summary = recipeData.summary;
    const cookTime = recipeData.readyInMinutes;
    const firstDishType = recipeData.dishTypes[0]; // Get the first dish type

    // Process recipe data further if needed
  })
  .catch(error => {
    console.error('Error fetching recipe data:', error);
  });

// Fetch ingredient data
fetch('./spoonIngredientInfo.json')
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(ingredientData => {
    // Assuming ingredientData is an array of ingredients
    const pantryItem = ingredientData.map(ingredient => {
      return {
        aisle: ingredient.aisle, // Including aisle information
        name: ingredient.name,
        quantity: ingredient.quantity, // Assuming you have quantity in your JSON data
        expiration: ingredient.expiration, // Assuming you have expiration date in your JSON data
        nutrients: ingredient.nutrition // Assuming nutrition is structured as expected in your JSON data
      };
    });

    console.log(pantryItem);
  })
  .catch(error => {
    console.error('Error fetching ingredient data:', error);
  });

