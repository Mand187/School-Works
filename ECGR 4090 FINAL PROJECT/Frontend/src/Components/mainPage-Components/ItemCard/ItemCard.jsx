import React from 'react';
import './ItemCard.css'; // Add styles as needed

const ItemCard = ({ item }) => {
  return (
    <div className="item-card">
      <img src={item.image} alt={item.name} />
      <h4>{item.name}</h4>
      <p>{item.quantity}</p>
    </div>
  );
};

export default ItemCard;
