import React from 'react';
import './loginStyle.css';

const LoginPage = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    // Handle the form submission logic here
    console.log('Form submitted');
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
          required
          aria-label="Username"
        />
        
        <label htmlFor="psw">Password</label>
        <input
          type="password"
          id="psw"
          placeholder="Enter Password"
          name="psw"
          required
          aria-label="Password"
        />
        
        <button type="submit" id="loginButton">Login</button>
      </form>
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
