import React from 'react';
import { Box, Typography, Button, Grid, Paper } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import TimelineIcon from '@mui/icons-material/Timeline';
import ScienceIcon from '@mui/icons-material/Science';
import SchoolIcon from '@mui/icons-material/School';

const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Make Predictions',
      description: 'Use our trained model to predict obesity risk levels based on patient data.',
      icon: <ScienceIcon sx={{ fontSize: 40, color: '#FFD700' }} />,
      path: '/predict'
    },
    {
      title: 'Visualize Data',
      description: 'Explore interactive visualizations of obesity risk factors and their relationships.',
      icon: <TimelineIcon sx={{ fontSize: 40, color: '#FFD700' }} />,
      path: '/visualize'
    },
    {
      title: 'Train Model',
      description: 'Upload new data to retrain the model and improve its prediction accuracy.',
      icon: <SchoolIcon sx={{ fontSize: 40, color: '#FFD700' }} />,
      path: '/train'
    }
  ];

  return (
    <Box sx={{ bgcolor: '#000000', minHeight: '100vh', p: 3 }}>
      <Box sx={{ textAlign: 'center', mb: 8, mt: 4 }}>
        <Typography variant="h3" component="h1" sx={{ color: '#FFD700', fontWeight: 'bold', mb: 2 }}>
          ML Pipeline for Obesity Risk Prediction
        </Typography>
        <Typography variant="h6" sx={{ color: '#FFD700', opacity: 0.8, maxWidth: 800, mx: 'auto' }}>
          A machine learning solution for predicting obesity risk levels using patient data and lifestyle factors
        </Typography>
      </Box>

      <Grid container spacing={4} justifyContent="center">
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Paper
              sx={{
                p: 4,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                bgcolor: '#000000',
                border: '1px solid #FFD700',
                borderRadius: 2,
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: '0 4px 20px rgba(255, 215, 0, 0.2)',
                },
              }}
            >
              <Box sx={{ mb: 2 }}>{feature.icon}</Box>
              <Typography variant="h5" component="h2" sx={{ color: '#FFD700', mb: 2, fontWeight: 'bold' }}>
                {feature.title}
              </Typography>
              <Typography variant="body1" sx={{ color: '#FFD700', opacity: 0.8, mb: 3, flexGrow: 1 }}>
                {feature.description}
              </Typography>
              <Button
                variant="contained"
                onClick={() => navigate(feature.path)}
                sx={{
                  bgcolor: '#000000',
                  color: '#FFD700',
                  border: '2px solid #FFD700',
                  '&:hover': {
                    bgcolor: '#FFD700',
                    color: '#000000',
                  },
                  px: 4,
                  py: 1,
                  fontSize: '1.1rem',
                }}
              >
                Get Started
              </Button>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Home; 