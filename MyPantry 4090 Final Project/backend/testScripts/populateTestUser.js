const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const userData = require('../models/userLoginModel'); // Adjust the path if necessary

// Connect to MongoDB
mongoose.connect("mongodb://127.0.0.1:27017/myPantry", { useNewUrlParser: true, useUnifiedTopology: true })
    .then(async () => {
        console.log('Connected to MongoDB successfully');
        
        // Create a test user
        const testUser = {
            username: 'testuser',
            password: await bcrypt.hash('password123', 10), // Hash the password
            email: 'testuser@example.com'
        };

        try {
            // Save the test user to the database
            const user = new userData(testUser);
            await user.save();
            console.log('Test user created successfully');
        } catch (err) {
            console.error('Error creating test user:', err);
        } finally {
            // Close the connection
            mongoose.connection.close();
        }
    })
    .catch(err => {
        console.error('Error connecting to MongoDB:', err);
    });
