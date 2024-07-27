import React from 'react';
import './CategoryCard.css'; // Add styles as needed

const CategoryCard = ({ category }) => {
  return (
    <div className="category-card">
      <h3>{category.name}</h3>
      <div className="card-items">
        {category.items.map((item) => (
          <div className="card-item" key={item.id}>
            <img src={item.image} alt={item.name} />
            <p>{item.name}</p>
          </div>
        ))}
      </div>
      <p className="item-count">Total Items: {category.items.length}</p>
    </div>
  );
};

export default CategoryCard;
