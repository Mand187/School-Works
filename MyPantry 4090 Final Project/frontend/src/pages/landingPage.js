import React, { useState } from 'react';
import './styles/LandingPage.css'; // Updated import

function LandingPage() {
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle('dark-mode', !darkMode);
  };

  return (
    <div className={`landing-page ${darkMode ? 'dark' : 'light'}`}>
      <header className="top-bar">
        <div className="app-name">myPantry</div>
        <nav className="nav-links">
          <a href="/">Home</a>
          <a href="/about">About</a>
          <a href="/login">Login</a>
        </nav>
        <button className="dark-mode-toggle" onClick={toggleDarkMode}>
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
      </header>
      <main className="content">
        <h1 className="project-title">Welcome to myPantry</h1>
        <p className="project-description">Your personalized culinary assistant. Organize your recipes, plan your meals, and manage your pantry easily.</p>
        <button className="get-started-button">Get Started</button>
      </main>
    </div>
  );
}

export default LandingPage;
