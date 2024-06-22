// models/connectionLogModel.js
const mongoose = require("mongoose");

// Define a schema for the connection log
const connectionLogSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now }
});

// Create a model for the connection log
const ConnectionLog = mongoose.model('ConnectionLog', connectionLogSchema);

module.exports = ConnectionLog;
