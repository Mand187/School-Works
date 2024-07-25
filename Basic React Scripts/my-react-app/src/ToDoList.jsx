import React, { useEffect, useState } from 'react';

function ToDoList() {
    const [tasks, setTasks] = useState([]);
    const [newTask, setNewTask] = useState("");

    useEffect(() => {
        fetchTasks();
    }, []);

    const fetchTasks = async () => {
        try {
            const response = await fetch('http://localhost:5000/getTasks');
            const data = await response.json();
            setTasks(data);
        } catch (error) {
            console.error('Error fetching tasks:', error);
        }
    };

    function handleInputChange(event) {
        setNewTask(event.target.value);
    }

    function addTask() {
        if (newTask.trim() !== "") {
            setTasks(t => [...t, { task: newTask }]);  // Add as an object
            setNewTask("");
        }
    }

    function clearTasks() {
        setTasks([]);
    }

    async function saveTasks() {
        try {
            // Ensure tasks is a valid array of objects
            if (!Array.isArray(tasks) || tasks.some(task => typeof task !== 'object')) {
                throw new Error('Invalid data format');
            }

            // Send only the relevant data
            const response = await fetch('http://localhost:5000/saveTasks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(tasks)  // Make sure tasks is the correct format
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            console.log('Tasks saved:', result);
        } catch (error) {
            console.error('Error saving tasks:', error);
        }
    }

    function deleteTask(index) {
        const updatedTasks = tasks.filter((_, i) => i !== index);
        setTasks(updatedTasks);
    }

    function moveTaskUp(index) {
        if (index > 0) {
            const updatedTasks = [...tasks];
            [updatedTasks[index], updatedTasks[index - 1]] = [updatedTasks[index - 1], updatedTasks[index]];
            setTasks(updatedTasks);
        }
    }

    function moveTaskDown(index) {
        if (index < tasks.length - 1) {
            const updatedTasks = [...tasks];
            [updatedTasks[index], updatedTasks[index + 1]] = [updatedTasks[index + 1], updatedTasks[index]];
            setTasks(updatedTasks);
        }
    }

    return (
        <div className="to-do-list">
            <h1>To-Do List</h1>
            <div className="input-container">
                <input
                    type='text'
                    placeholder='Enter a task...'
                    value={newTask}
                    onChange={handleInputChange}
                />
                <button className='add-button' onClick={addTask}>Add</button>
                <button className='clear-button' onClick={clearTasks}>Clear</button>
            </div>
            <button className='save-button' onClick={saveTasks}>Push to Database</button>
            <ol>
                {tasks.map((task, index) =>
                    <li key={index}>
                        <span className='text'>{task.task}</span>  {/* Access task text */}
                        <button className='delete-button' onClick={() => deleteTask(index)}>DELETE</button>
                        <button className='move-button' onClick={() => moveTaskUp(index)}>UP</button>
                        <button className='move-button' onClick={() => moveTaskDown(index)}>DOWN</button>
                    </li>
                )}
            </ol>
        </div>
    );
}

export default ToDoList;
