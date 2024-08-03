const express = require('express');
const router = express.Router();
const { getPantryItems, addPantryItem, deletePantryItem } = require('../controllers/pantryController');

// Define routes with callback functions
router.route('/')
  .get(getPantryItems) // Ensure getPantryItems is defined in pantryController
  .post(addPantryItem); // Ensure addPantryItem is defined in pantryController

router.route('/:id')
  .delete(deletePantryItem); // Ensure deletePantryItem is defined in pantryController

module.exports = router;
