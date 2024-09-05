// models/User.js
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  token: { type: String }, // Add the token field
  preferences: {
    dietaryRestrictions: [String],
    favoriteRecipes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Recipe' }],
  },
}, { timestamps: true });

const User = mongoose.model('User', userSchema);



