import React from 'react';
import { Navigate } from 'react-router-dom';

// Utility function to check if token is valid (implement your own logic here)
const isAuthenticated = () => {
  const token = localStorage.getItem('token');
  // Check if token exists and is valid (you can add more checks if needed)
  return !!token;
};

const AuthRoute = ({ element }) => {
  return isAuthenticated() ? element : <Navigate to="/login" />;
};

export default AuthRoute; 
