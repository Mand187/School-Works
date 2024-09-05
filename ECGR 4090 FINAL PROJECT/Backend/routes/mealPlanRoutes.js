// mealPlanRoutes.js
const express = require('express');
const router = express.Router();
const { MealPlan } = require('../models');
const auth = require('../middleware/auth');

// Get meal plan for a specific date
router.get('/:date', auth, async (req, res) => {
  try {
    const date = new Date(req.params.date);
    const mealPlan = await MealPlan.findOne({ user: req.user.id, date }).populate('meals.recipe');
    res.json(mealPlan);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Create or update meal plan for a specific date
router.post('/:date', auth, async (req, res) => {
  try {
    const date = new Date(req.params.date);
    let mealPlan = await MealPlan.findOne({ user: req.user.id, date });
    
    if (mealPlan) {
      mealPlan.meals = req.body.meals;
    } else {
      mealPlan = new MealPlan({
        user: req.user.id,
        date,
        meals: req.body.meals
      });
    }
    
    await mealPlan.save();
    res.json(mealPlan);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

module.exports = router;