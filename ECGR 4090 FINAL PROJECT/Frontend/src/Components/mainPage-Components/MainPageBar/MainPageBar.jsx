import React, { useState, useEffect } from 'react';
import CategoryCard from '../CategoryCard/CategoryCard';
import './MainPageBar.css'; // Ensure the path is correct

const MainPageBar = () => {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:3000/pantry');
        if (!response.ok) {
          throw new Error('Failed to fetch categories');
        }
        const data = await response.json();
        setCategories(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="pantry-container">
      <div className="Row">
        <button className="pantryBtn" onClick={() => window.location.href = '/pantry'}>Pantry</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/recipe'}>Recipe</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/shopping-list'}>Shopping List</button>
        <button className="pantryBtn" onClick={() => window.location.href = '/cook-now'}>Cook Now</button>
      </div>
      
      {categories.map((category) => (
        <CategoryCard key={category.category} category={category} />
      ))}
    </div>
  );
};

export default MainPageBar;
