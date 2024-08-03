import React, { useState, useEffect } from 'react';
import './MainPageBar.css'; // Ensure the path is correct

const MainPageBar = () => {

  return (
    <div className="pantry-container">
      <div className="Row">
        <button className="pantryBtn" onClick={() => window.location.href = '/pantry'}>Pantry</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/recipe'}>Recipe</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/shopping-list'}>Shopping</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/cook-now'}>Cook Now</button>
      </div>
    </div>
  );
};

export default MainPageBar;
