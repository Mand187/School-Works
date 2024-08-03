import React, { useState, useEffect, } from 'react';
import './MPNavBar.css';

const MPNavBar = () => {
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
      <div className='usrWelcome'> Hello, Username goes here</div>
      <ul>
        <li><a href="#home" className='navbar-Btn'>Home</a></li>
        <li><a href="#Help" className='navbar-Btn'>Help</a></li>
        <li><a href="#Settings" className='navbar-Btn'>Settings</a></li>
        <li><a href="/" className='navbar-Btn'>Sign Out</a></li>
      </ul>
    </nav>
  );
}

export default MPNavBar;
