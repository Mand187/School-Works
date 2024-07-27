import React from 'react';
import Navbar from '../../Components/Home-Components/Navbar/Navbar';
import MainPageBar from '../../Components/mainPage-Components/MainPageBar/MainPageBar';

const MainPage = () => {
  return (
    <>
      <Navbar />
      <section id="pantry">
        <MainPageBar />
      </section>
    </>
  );
};

export default MainPage;
