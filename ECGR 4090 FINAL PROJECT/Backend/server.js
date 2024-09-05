require('dotenv').config(); // Load environment variables early
console.log('MONGO_URI:', process.env.MONGO_URI); // Log to check if it's loaded correctly

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const connectDB = require('./config/db');

// ... rest of your code


// Import models
const { FoodItem, Recipe, User, MealPlan, ShoppingList } = require('./models');

// Initialize express app
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet()); // Set security headers
app.use(express.json()); // Parse incoming JSON requests
app.use(cors()); // Enable CORS

// Connect to MongoDB using connectDB function
connectDB();

// Import routes
const userRoutes = require('./routes/userRoutes');
const pantryRoutes = require('./routes/pantryRoutes');
const mealPlanRoutes = require('./routes/mealPlanRoutes');
const shoppingListRoutes = require('./routes/shoppingListRoutes');

// Use routes
app.use('/api/users', userRoutes);
app.use('/api/pantry', pantryRoutes);
app.use('/api/meal-plans', mealPlanRoutes);
app.use('/api/shopping-lists', shoppingListRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: 'Something went wrong!', error: err.message });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app; // Export for testing purposes
