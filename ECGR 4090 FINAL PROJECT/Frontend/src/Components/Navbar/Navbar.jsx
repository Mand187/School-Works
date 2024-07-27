import React, { useState, useEffect } from 'react';
import './Navbar.css';

const Navbar = () => {
  const [sticky, setSticky] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setSticky(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <nav className={`container ${sticky ? 'sticky' : ''}`}>
      <div className='Title'>myPantry</div>
      <ul>
        <li><a href="#home" className='navbar-Btn'>Home</a></li>
        <li><a href="#about" className='navbar-Btn'>About</a></li>
        <li><a href="#contact" className='navbar-Btn'>Contact</a></li>
        <li><a href="#copyright" className='navbar-Btn'>Copyright</a></li>
      </ul>
    </nav>
  );
}

export default Navbar;
