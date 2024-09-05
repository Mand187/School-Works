// pantryRoutes.js
const express = require('express');
const router = express.Router();
const { FoodItem } = require('../models');
const auth = require('../middleware/auth');

// Get all food items for a user
router.get('/', auth, async (req, res) => {
  try {
    const foodItems = await FoodItem.find({ user: req.user.id });
    res.json(foodItems);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Add a new food item
router.post('/', auth, async (req, res) => {
  try {
    const foodItem = new FoodItem({
      ...req.body,
      user: req.user.id
    });
    await foodItem.save();
    res.status(201).json(foodItem);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Update a food item
router.put('/:id', auth, async (req, res) => {
  try {
    const foodItem = await FoodItem.findOneAndUpdate(
      { _id: req.params.id, user: req.user.id },
      req.body,
      { new: true }
    );
    if (!foodItem) {
      return res.status(404).json({ message: 'Food item not found' });
    }
    res.json(foodItem);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Delete a food item
router.delete('/:id', auth, async (req, res) => {
  try {
    const foodItem = await FoodItem.findOneAndDelete({ _id: req.params.id, user: req.user.id });
    if (!foodItem) {
      return res.status(404).json({ message: 'Food item not found' });
    }
    res.json({ message: 'Food item deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;