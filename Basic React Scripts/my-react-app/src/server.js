import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import bodyParser from 'body-parser';

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json());

const uri = 'mongodb://localhost:27017/todolist';

mongoose.connect(uri)
    .then(() => console.log('MongoDB connected...'))
    .catch(err => console.error('MongoDB connection error:', err));

const taskSchema = new mongoose.Schema({
    text: String,
    completed: Boolean
});

const Task = mongoose.model('Task', taskSchema);

app.post('/saveTasks', async (req, res) => {
    try {
        await Task.deleteMany({});
        await Task.insertMany(req.body.tasks);
        res.status(200).send('Tasks saved successfully');
    } catch (error) {
        console.error('Error saving tasks:', error);
        res.status(500).send('Error saving tasks');
    }
});

app.get('/getTasks', async (req, res) => {
    try {
        const tasks = await Task.find({});
        res.status(200).json(tasks);
    } catch (error) {
        console.error('Error retrieving tasks:', error);
        res.status(500).send('Error retrieving tasks');
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
