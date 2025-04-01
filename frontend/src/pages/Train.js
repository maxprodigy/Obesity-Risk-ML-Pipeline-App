import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import ModelTraining from '../components/train/ModelTraining';

const Train = () => {
  return (
    <Box sx={{ bgcolor: '#000000', minHeight: '100vh', p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#FFD700', fontWeight: 'bold' }}>
        Model Training
      </Typography>
      <Typography variant="body1" sx={{ color: '#FFD700', mb: 4, opacity: 0.8 }} paragraph>
        Upload new training data to retrain the model and improve its prediction accuracy.
      </Typography>
      <Paper elevation={0} sx={{ bgcolor: 'transparent' }}>
        <ModelTraining />
      </Paper>
    </Box>
  );
};

export default Train; 