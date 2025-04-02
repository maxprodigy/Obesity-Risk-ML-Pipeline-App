import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ScatterController
} from 'chart.js';
import { Bar, Line, Scatter } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ScatterController
);

const DataVisualization = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visualizationData, setVisualizationData] = useState(null);

  useEffect(() => {
    fetchVisualizationData();
  }, []);

  const fetchVisualizationData = async () => {
    try {
      console.log('Fetching visualization data...');
      const response = await fetch(`${process.env.REACT_APP_API_URL}/visualize/data`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch visualization data');
      }
      
      const data = await response.json();
      console.log('Received visualization data:', data);
      
      if (!data || Object.keys(data).length === 0) {
        throw new Error('No visualization data available');
      }
      
      setVisualizationData(data);
      setError(null);
    } catch (err) {
      console.error('Visualization error:', err);
      setError(err.message || 'Failed to fetch visualization data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#FFD700',
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#FFD700',
        bodyColor: '#FFD700',
        borderColor: '#FFD700',
        borderWidth: 1,
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 215, 0, 0.1)',
        },
        ticks: {
          color: '#FFD700'
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 215, 0, 0.1)',
        },
        ticks: {
          color: '#FFD700',
          callback: function(value) {
            if (this.chart.data.datasets[0].label.includes('Risk')) {
              return (value * 100).toFixed(0) + '%';
            }
            return value;
          }
        }
      }
    }
  };

  const pieChartOptions = {
    ...chartOptions,
    scales: undefined,
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress sx={{ color: '#FFD700' }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert 
        severity="error"
        sx={{
          bgcolor: 'transparent',
          color: '#FFD700',
          border: '1px solid #FFD700',
          '& .MuiAlert-icon': {
            color: '#FFD700',
          },
        }}
      >
        {error}
      </Alert>
    );
  }

  const riskDistributionData = visualizationData?.risk_distribution ? {
    labels: visualizationData.risk_distribution.labels,
    datasets: [{
      label: 'Number of Cases',
      data: visualizationData.risk_distribution.data,
      backgroundColor: ['rgba(255, 215, 0, 0.6)', 'rgba(255, 165, 0, 0.6)', 'rgba(218, 165, 32, 0.6)'],
      borderColor: ['#FFD700', '#FFA500', '#DAA520'],
      borderWidth: 2,
    }],
  } : null;

  const bmiCorrelationData = visualizationData?.bmi_correlation ? {
    labels: visualizationData.bmi_correlation.labels,
    datasets: [{
      label: 'Risk Probability',
      data: visualizationData.bmi_correlation.data,
      borderColor: '#FFD700',
      backgroundColor: 'rgba(255, 215, 0, 0.1)',
      fill: true,
      tension: 0.4,
      pointBackgroundColor: '#000000',
      pointBorderColor: '#FFD700',
      pointBorderWidth: 2,
      pointRadius: 4,
    }],
  } : null;

  const ageDistributionData = visualizationData?.age_distribution ? {
    labels: visualizationData.age_distribution.labels,
    datasets: [{
      label: 'Risk by Age Group',
      data: visualizationData.age_distribution.data,
      backgroundColor: 'rgba(255, 215, 0, 0.6)',
      borderColor: '#FFD700',
      borderWidth: 2,
    }],
  } : null;

  const genderDistributionData = visualizationData?.gender_distribution ? {
    labels: visualizationData.gender_distribution.labels,
    datasets: [{
      label: 'Risk by Gender',
      data: visualizationData.gender_distribution.data,
      backgroundColor: ['rgba(255, 215, 0, 0.6)', 'rgba(255, 165, 0, 0.6)'],
      borderColor: ['#FFD700', '#FFA500'],
      borderWidth: 2,
    }],
  } : null;

  const scatterOptions = {
    ...chartOptions,
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 215, 0, 0.1)',
        },
        ticks: {
          color: '#FFD700'
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 215, 0, 0.1)',
        },
        ticks: {
          color: '#FFD700'
        },
        title: {
          display: true,
          text: 'Number of Cases',
          color: '#FFD700'
        }
      }
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={3}>
        {riskDistributionData && (
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                bgcolor: '#000000',
                border: '1px solid #FFD700',
                borderRadius: 2,
                height: 400,
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ color: '#FFD700', textAlign: 'center' }}>
                Risk Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar
                  data={riskDistributionData}
                  options={chartOptions}
                />
              </Box>
            </Paper>
          </Grid>
        )}
        {bmiCorrelationData && (
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                bgcolor: '#000000',
                border: '1px solid #FFD700',
                borderRadius: 2,
                height: 400,
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ color: '#FFD700', textAlign: 'center' }}>
                BMI vs Risk Correlation
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line
                  data={bmiCorrelationData}
                  options={chartOptions}
                />
              </Box>
            </Paper>
          </Grid>
        )}
        {ageDistributionData && (
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                bgcolor: '#000000',
                border: '1px solid #FFD700',
                borderRadius: 2,
                height: 400,
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ color: '#FFD700', textAlign: 'center' }}>
                Risk by Age Group
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar
                  data={ageDistributionData}
                  options={chartOptions}
                />
              </Box>
            </Paper>
          </Grid>
        )}
        {genderDistributionData && (
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                bgcolor: '#000000',
                border: '1px solid #FFD700',
                borderRadius: 2,
                height: 400,
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ color: '#FFD700', textAlign: 'center' }}>
                Risk by Gender
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar
                  data={genderDistributionData}
                  options={chartOptions}
                />
              </Box>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default DataVisualization; 
