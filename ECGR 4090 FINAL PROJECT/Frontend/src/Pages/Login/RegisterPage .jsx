import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './loginStyle.css'; // Use same or different styles

const generateRandomToken = () => {
  return Math.random().toString(36).substr(2, 16); // Generate a random token
};

const RegisterPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate(); // Initialize navigate

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await fetch('http://localhost:3000/users/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('Registration failed');
      }

      const data = await response.json();
      console.log('Registration successful', data);

      // Generate and store a random token
      const token = generateRandomToken();
      console.log('Generated Token:', token);
      localStorage.setItem('token', token); // Store the token in localStorage

      // Redirect to login or main page
      navigate('/login'); // Redirect to login page

    } catch (error) {
      console.error('Error:', error);
      setErrorMessage('Registration failed. Please try again.');
    }
  };

  return (
    <div className="login-container">
      <h2>Register</h2>
      <form id="registerForm" onSubmit={handleSubmit}>
        <label htmlFor="uname">Username</label>
        <input
          type="text"
          id="uname"
          placeholder="Enter Username"
          name="uname"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          aria-label="Username"
        />
        
        <label htmlFor="psw">Password</label>
        <input
          type="password"
          id="psw"
          placeholder="Enter Password"
          name="psw"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          aria-label="Password"
        />
        
        <button type="submit" id="loginButton">Register</button>
      </form>
      {errorMessage && <p className="error-message">{errorMessage}</p>}
    </div>
  );
};

export default RegisterPage;
