import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import DataVisualization from '../components/visualize/DataVisualization';

const Visualize = () => {
  return (
    <Box sx={{ bgcolor: '#000000', minHeight: '100vh', p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#FFD700', fontWeight: 'bold' }}>
        Data Visualization
      </Typography>
      <Typography variant="body1" sx={{ color: '#FFD700', mb: 4, opacity: 0.8 }} paragraph>
        Explore interactive visualizations of obesity risk factors and their relationships through comprehensive charts and graphs.
      </Typography>
      <Paper elevation={0} sx={{ bgcolor: 'transparent' }}>
        <DataVisualization />
      </Paper>
    </Box>
  );
};

export default Visualize; 