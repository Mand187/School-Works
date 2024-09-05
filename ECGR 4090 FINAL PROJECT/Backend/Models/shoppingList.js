const shoppingListSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  items: [{
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'FoodItem' },
    quantity: Number,
    unit: String,
    isPurchased: { type: Boolean, default: false },
  }],
  name: { type: String, default: 'My Shopping List' },
}, { timestamps: true });

const ShoppingList = mongoose.model('ShoppingList', shoppingListSchema);
