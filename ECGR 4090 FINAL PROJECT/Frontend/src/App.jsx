import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Pages/Landing/Landing'; 
import LoginPage from './Pages/Login/loginPage'; 
import RegisterPage from './Pages/Login/RegisterPage ';
import MainPage from './Pages/MainPage/MainPage'; // Import the MainPage component
import AuthRoute from './Components/AuthRoute';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/main" element={<AuthRoute element={<MainPage />} />} />
        {/* Add other routes here */}
      </Routes>
    </Router>
  );
};

export default App;
