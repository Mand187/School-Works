import React from 'react';
import { Link } from 'react-router-dom';
import './Hero.css';

const Hero = () => {
  return (
    <div className='hero'>
      <div className='hero-text'>
        <h1>Welcome to myPantry!</h1>
        <p>Your personalized culinary assistant. Organize your recipes, plan your meals, and manage your pantry easily.</p>
        <Link to="/login" className='defaultBtn'>Get Started</Link>
      </div>
    </div>
  );
};

export default Hero;
