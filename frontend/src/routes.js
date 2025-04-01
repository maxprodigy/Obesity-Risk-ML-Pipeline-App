import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Visualize from './pages/Visualize';
import Train from './pages/Train';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
      <Route path="/visualize" element={<Visualize />} />
      <Route path="/train" element={<Train />} />
    </Routes>
  );
};

export default AppRoutes; 