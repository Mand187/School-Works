const foodItemSchema = new mongoose.Schema({
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

const FoodItem = mongoose.model('FoodItem', foodItemSchema);

