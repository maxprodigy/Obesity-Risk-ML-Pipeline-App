import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './theme';
import Navbar from './components/Navbar';
import AppRoutes from './routes';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div style={{ backgroundColor: '#000000', minHeight: '100vh' }}>
          <Navbar />
          <AppRoutes />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App; 