const express = require('express');
const router = express.Router();
const User = require('../models/user');

// Create a new user
router.post('/create', async (req, res) => {
  try {
    const { username, password } = req.body;
    const user = new User({ username, password });
    await user.save();
    res.status(201).json(user);
  } catch (error) {
    if (error.code === 11000) {
      // Duplicate key error
      return res.status(400).json({ message: 'Username already exists' });
    }
    res.status(500).json({ message: 'Error creating user', error });
  }
});

// Get all users
router.get('/all', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching users', error });
  }
});

module.exports = router;
