import React from 'react';
import './loginStyle.css'; // Use same or different styles

const RegisterPage = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    // Handle registration form submission
    console.log('Registration form submitted');
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
        
        <button type="submit" id="loginButton">Register</button>
      </form>
    </div>
  );
};

export default RegisterPage;
