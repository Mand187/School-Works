const PantryItem = require('../models/PantryItem');

// Get all pantry items
const getPantryItems = async (req, res) => {
  try {
    const items = await PantryItem.find();
    res.json(items);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Add a new pantry item
const addPantryItem = async (req, res) => {
  const { name, quantity, expirationDate } = req.body;

  try {
    const newItem = new PantryItem({ name, quantity, expirationDate });
    const savedItem = await newItem.save();
    res.status(201).json(savedItem);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Delete a pantry item
const deletePantryItem = async (req, res) => {
  try {
    const item = await PantryItem.findById(req.params.id);
    if (!item) {
      return res.status(404).json({ message: 'Item not found' });
    }
    await item.remove();
    res.json({ message: 'Item removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { getPantryItems, addPantryItem, deletePantryItem };
