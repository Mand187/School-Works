// shoppingListRoutes.js
const express = require('express');
const router = express.Router();
const { ShoppingList, MealPlan, Recipe, FoodItem } = require('../models');
const auth = require('../middleware/auth');

// Get current shopping list
router.get('/', auth, async (req, res) => {
  try {
    let shoppingList = await ShoppingList.findOne({ user: req.user.id }).populate('items.item');
    if (!shoppingList) {
      shoppingList = new ShoppingList({ user: req.user.id, items: [] });
      await shoppingList.save();
    }
    res.json(shoppingList);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Update shopping list
router.put('/', auth, async (req, res) => {
  try {
    const shoppingList = await ShoppingList.findOneAndUpdate(
      { user: req.user.id },
      { items: req.body.items },
      { new: true, upsert: true }
    ).populate('items.item');
    res.json(shoppingList);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Generate shopping list based on meal plan
router.post('/generate', auth, async (req, res) => {
  try {
    const { startDate, endDate } = req.body;
    const mealPlans = await MealPlan.find({
      user: req.user.id,
      date: { $gte: new Date(startDate), $lte: new Date(endDate) }
    }).populate('meals.recipe');

    const neededIngredients = new Map();

    for (const mealPlan of mealPlans) {
      for (const meal of mealPlan.meals) {
        if (meal.recipe) {
          for (const ingredient of meal.recipe.ingredients) {
            const currentAmount = neededIngredients.get(ingredient.item.toString()) || 0;
            neededIngredients.set(ingredient.item.toString(), currentAmount + ingredient.quantity);
          }
        }
      }
    }

    const userFoodItems = await FoodItem.find({ user: req.user.id });
    const shoppingListItems = [];

    for (const [itemId, neededAmount] of neededIngredients) {
      const userItem = userFoodItems.find(item => item._id.toString() === itemId);
      if (!userItem || userItem.quantity < neededAmount) {
        shoppingListItems.push({
          item: itemId,
          quantity: neededAmount - (userItem ? userItem.quantity : 0),
          unit: userItem ? userItem.unit : 'units'
        });
      }
    }

    const shoppingList = await ShoppingList.findOneAndUpdate(
      { user: req.user.id },
      { items: shoppingListItems },
      { new: true, upsert: true }
    ).populate('items.item');

    res.json(shoppingList);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

module.exports = router;