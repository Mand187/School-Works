import React from 'react';
import Navbar from '../../Components/Home-Components/Navbar/Navbar';
import Hero from '../../Components/Home-Components/Hero/Hero';
import About from '../../Components/Home-Components/About/About';
import Contact from '../../Components/Home-Components/Contact/Contact';
import Copyright from '../../Components/Home-Components/Copyright/Copyright';
const Home = () => {
  return (
    <>
      <Navbar />
      <section id="home">
        <Hero />
      </section>
      <section id="about">
        <About />
      </section>
      <section id="contact">
        <Contact />
      </section>
      <section id="copyright">
        <Copyright />
      </section>
    </>
  );
};

export default Home;
