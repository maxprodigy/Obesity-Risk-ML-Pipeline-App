import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';

const Navbar = () => {
  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            textDecoration: 'none',
            color: '#FFD700',
            fontWeight: 'bold',
            flexGrow: 1,
          }}
        >
          ML Pipeline
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            component={RouterLink}
            to="/predict"
            sx={{
              color: '#FFD700',
              '&:hover': {
                bgcolor: 'rgba(255, 215, 0, 0.1)',
              },
            }}
          >
            Predict
          </Button>
          <Button
            component={RouterLink}
            to="/visualize"
            sx={{
              color: '#FFD700',
              '&:hover': {
                bgcolor: 'rgba(255, 215, 0, 0.1)',
              },
            }}
          >
            Visualize
          </Button>
          <Button
            component={RouterLink}
            to="/train"
            sx={{
              color: '#FFD700',
              '&:hover': {
                bgcolor: 'rgba(255, 215, 0, 0.1)',
              },
            }}
          >
            Train
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 