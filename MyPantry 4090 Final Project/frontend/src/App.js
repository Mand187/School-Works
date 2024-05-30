import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LoginPage from './pages/loginPage';
import RegisterPage from './pages/registerPage';
import Dashboard from './pages/Dashboard';
import LandingPage from './pages/landingPage';  // Correctly import the LandingPage component

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/"           element={<LandingPage  />}  />  
          <Route path="/login"      element={<LoginPage    />}  />
          <Route path="/register"   element={<RegisterPage />}  />
          <Route path="/dashboard"  element={<Dashboard    />}  />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
