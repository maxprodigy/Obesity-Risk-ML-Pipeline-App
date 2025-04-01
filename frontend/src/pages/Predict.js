import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import PredictionForm from '../components/predict/PredictionForm';

const Predict = () => {
  return (
    <Box sx={{ bgcolor: '#000000', minHeight: '100vh', p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#FFD700', fontWeight: 'bold' }}>
        Obesity Risk Prediction
      </Typography>
      <Typography variant="body1" sx={{ color: '#FFD700', mb: 4, opacity: 0.8 }} paragraph>
        Enter patient information to predict their obesity risk level using our machine learning model.
      </Typography>
      <Paper elevation={0} sx={{ bgcolor: 'transparent' }}>
        <PredictionForm />
      </Paper>
    </Box>
  );
};

export default Predict; 