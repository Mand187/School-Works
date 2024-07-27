const express = require('express');
const crypto = require('crypto'); // For token generation
const router = express.Router();
const User = require('../models/User');

// Registration route
router.post('/register', async (req, res) => {
  try {
    const { username, password } = req.body;

    // Generate a random token
    const token = crypto.randomBytes(16).toString('hex');

    // Create a new user with the generated token
    const user = new User({ username, password, token });
    await user.save();

    res.status(201).json({ message: 'User created successfully', token });
  } catch (error) {
    if (error.code === 11000) {
      return res.status(400).json({ message: 'Username already exists' });
    }
    res.status(500).json({ message: 'Error creating user', error });
  }
});

// Login route
router.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    const user = await User.findOne({ username });

    if (!user) {
      return res.status(401).json({ message: 'Invalid username or password' });
    }

    // Check if the passwords match
    if (password === user.password) {
      res.status(200).json({ message: 'Login successful', token: user.token });
    } else {
      res.status(401).json({ message: 'Invalid username or password' });
    }
  } catch (error) {
    console.error('Error during login:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
