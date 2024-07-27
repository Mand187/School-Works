import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './loginStyle.css';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate(); // Initialize navigate

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await fetch('http://localhost:3000/users/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Login failed:', errorData.message);
        throw new Error('Login failed');
      }

      const data = await response.json();
      console.log('Login successful');
      console.log('Response Data:', data);

      if (data.token) {
        console.log('JWT Token:', data.token);
        localStorage.setItem('token', data.token);
      }

      // Redirect to the main page upon successful login
      navigate('/main'); // Use navigate to redirect to /main

    } catch (error) {
      console.error('Error:', error);
      setErrorMessage('Login failed. Please check your credentials and try again.');
    }
  };

  return (
    <div className="login-container">
      <h2>Login</h2>
      <form id="loginForm" onSubmit={handleSubmit}>
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
        
        <button type="submit" id="loginButton">Login</button>
      </form>
      {errorMessage && <p className="error-message">{errorMessage}</p>}
      <div className="link-container">
        <div className="register-link">
          <a href="/register">Register</a>
        </div>
        <div className="password-reset-link">
          <a href="/forgot-password">Forgot password?</a>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
