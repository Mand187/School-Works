import React from 'react';
import MPNavBar from '../../Components/mainPage-Components/MaingPageNavBar/MPNavBar'
import MainPageBar from '../../Components/mainPage-Components/MainPageBar/MainPageBar';

const MainPage = () => {
  return (
    <>
      <MPNavBar />
      <section id="pantry">
        <MainPageBar />
      </section>
    </>
  );
};

export default MainPage;
