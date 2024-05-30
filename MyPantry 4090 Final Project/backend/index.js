const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const userData = require('./models/userLoginModel');
const ConnectionLog = require('./models/connectionLogModel');

const app = express();

app.use(express.json());
app.use(cors());

mongoose.connect("mongodb://127.0.0.1:27017/myPantry");

// Check for connection success
mongoose.connection.on('connected', async () => {
    console.log('Connected to MongoDB successfully');
    
    // Verify the database name
    const dbName = mongoose.connection.name;
    if (dbName === 'myPantry') {
        console.log('Successfully connected to the myPantry database');
    } else {
        console.log(`Connected to the wrong database: ${dbName}`);
    }
    
    // Create a new connection log entry
    const logEntry = new ConnectionLog();
    try {
        await logEntry.save();
        console.log('Connection log entry added');
    } catch (err) {
        console.log('Error adding connection log entry:', err);
    }
});

// Check for connection error
mongoose.connection.on('error', (err) => {
    console.log('Error connecting to MongoDB:', err);
});

app.listen(3000, () => {
    console.log("Server is running on port 3000");
});

app.post('/register', (req, res) => {
    userData.create(req.body)
        .then(myPantry => res.json(myPantry))
        .catch(err => res.json(err));
});
