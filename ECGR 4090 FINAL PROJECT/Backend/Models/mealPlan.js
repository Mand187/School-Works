const mealPlanSchema = new mongoose.Schema({
    user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    date: { type: Date, required: true },
    meals: [{
      type: { type: String, enum: ['breakfast', 'lunch', 'dinner', 'snack'] },
      recipe: { type: mongoose.Schema.Types.ObjectId, ref: 'Recipe' },
    }],
  }, { timestamps: true });
  
  const MealPlan = mongoose.model('MealPlan', mealPlanSchema);
