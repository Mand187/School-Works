const mongoose = require('mongoose');

const FoodItemSchema = new mongoose.Schema({
  name: { type: String, required: true },
  quantity: { type: Number, required: true },
  unit: { type: String, required: true },
  expirationDate: { type: Date },
  nutritionalInfo: {
    calories: Number,
    protein: Number,
    carbs: Number,
    fat: Number,
  },
  barcode: { type: String, unique: true },
  category: { type: String },
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
}, { timestamps: true });

const RecipeSchema = new mongoose.Schema({
  name: { type: String, required: true },
  ingredients: [{
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'FoodItem' },
    quantity: Number,
    unit: String,
  }],
  instructions: [String],
  prepTime: Number,
  cookTime: Number,
  servings: Number,
  tags: [String],
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
}, { timestamps: true });

const UserSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  preferences: {
    dietaryRestrictions: [String],
    favoriteRecipes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Recipe' }],
  },
}, { timestamps: true });

const MealPlanSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  date: { type: Date, required: true },
  meals: [{
    type: { type: String, enum: ['breakfast', 'lunch', 'dinner', 'snack'] },
    recipe: { type: mongoose.Schema.Types.ObjectId, ref: 'Recipe' },
  }],
}, { timestamps: true });

const ShoppingListSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  items: [{
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'FoodItem' },
    quantity: Number,
    unit: String,
    isPurchased: { type: Boolean, default: false },
  }],
  name: { type: String, default: 'My Shopping List' },
}, { timestamps: true });

const FoodItem = mongoose.model('FoodItem', FoodItemSchema);
const Recipe = mongoose.model('Recipe', RecipeSchema);
const User = mongoose.model('User', UserSchema);
const MealPlan = mongoose.model('MealPlan', MealPlanSchema);
const ShoppingList = mongoose.model('ShoppingList', ShoppingListSchema);

module.exports = {
  FoodItem,
  Recipe,
  User,
  MealPlan,
  ShoppingList
};