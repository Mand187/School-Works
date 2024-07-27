import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Pages/Landing/Landing'; 
import LoginPage from './Pages/Login/loginPage'; 
import RegisterPage from './Pages/Login/RegisterPage ';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        {/* Add other routes here */}
      </Routes>
    </Router>
  );
};

export default App;
