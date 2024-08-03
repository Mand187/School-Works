require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const helmet = require('helmet');
const connectDB = require('./config/db');

// Initialize express app
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet()); // Use Helmet early to set security headers
app.use(bodyParser.json());
app.use(cors());

// Connect to MongoDB using connectDB function
connectDB();

// Import routes
const userRoutes = require('./routes/userRoutes');
const pantryRoutes = require('./routes/pantryRoutes');

// Use routes
app.use('/users', userRoutes);
app.use('/api/pantry', pantryRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
