import React from 'react';
import './About.css';

const About = () => {
  return (
    <div className='About'>
      <div className='About-content'>
        <div className='AboutP'>
          <p>PLACE HOLDER FOR UNCC LOGO</p>
        </div>
        <h1>About myPantry</h1>
        <p>
          This represents a final project completed for ECGR 4090 Cloud Native Application Architecture in the Spring semester of 2024.
        </p>
        <h2>Features</h2>
        <ul className='features-list'>
          <li>Takes input of food items that a user has in their possession</li>
          <li>Keeps track of nutritional information</li>
          <li>Keeps track of expiration dates</li>
          <li>Allow users to scan barcodes</li>
          <li>Given the food items the user has, display recipes that a user could feasibly make</li>
          <li>Allow users to display favorites</li>
          <li>Select from categories</li>
          <li>Allow users to set what meals they would like to make</li>
          <li>Create shopping lists based on:</li>
          <ul className='sub-features-list'>
            <li>User input: What the user has designated as things they want</li>
            <li>Items required to make meals they wish to make</li>
          </ul>
          <li>Services must store information and run natively on the cloud</li>
        </ul>
      </div>
    </div>
  );
}

export default About;
