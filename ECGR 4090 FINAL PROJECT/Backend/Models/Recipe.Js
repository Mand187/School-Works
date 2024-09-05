const recipeSchema = new mongoose.Schema({
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
  
  const Recipe = mongoose.model('Recipe', recipeSchema);
