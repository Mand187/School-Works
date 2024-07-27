import React from 'react';
import './Hero.css';

const Hero = () => {
  return (
    <div className='hero'>
      <div className='hero-text'>
        <h1>Welcome to myPantry!</h1>
        <p>Your personalized culinary assistant. Organize your recipes, plan your meals, and manage your pantry easily.</p>
        <button className='defaultBtn'>Get Started</button>
      </div>
    </div>
  );
};

export default Hero;
