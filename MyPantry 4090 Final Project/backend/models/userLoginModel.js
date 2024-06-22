const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    email:    { type: String, required: true, unique: true },
});

const userData = mongoose.model("userData", userSchema);

module.exports = userData;
