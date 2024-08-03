const mongoose = require('mongoose');

const pantryItemSchema = mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  quantity: {
    type: Number,
    required: true,
  },
  expirationDate: {
    type: Date,
    required: true,
  },
  nutritionalInfo: {
    type: Object, // Can be detailed further based on your requirements
    required: false,
  },
}, {
  timestamps: true,
});

const PantryItem = mongoose.model('PantryItem', pantryItemSchema);

module.exports = PantryItem;
