import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Tab,
  Tabs,
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
  Legend
} from 'chart.js';
import { Bar, Line, Scatter } from 'react-chartjs-2';
import axios from 'axios';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const API_URL = 'http://localhost:8000';

const DataVisualization = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visualizationData, setVisualizationData] = useState(null);

  useEffect(() => {
    fetchVisualizationData();
  }, []);

  const fetchVisualizationData = async () => {
    try {
      const response = await axios.get(`${API_URL}/visualizations`);
      setVisualizationData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load visualization data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const renderRegionalDistribution = () => {
    if (!visualizationData?.regional) return null;

    const data = {
      labels: Object.keys(visualizationData.regional),
      datasets: [{
        label: 'Average Obesity Risk',
        data: Object.values(visualizationData.regional),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Obesity Risk by Region'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Average Risk Score'
          }
        }
      }
    };

    return <Bar data={data} options={options} />;
  };

  const renderHealthMetrics = () => {
    if (!visualizationData?.health_metrics) return null;

    const data = {
      datasets: [{
        label: 'BMI vs Obesity Risk',
        data: visualizationData.health_metrics.map(d => ({
          x: d.bmi,
          y: d.risk
        })),
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'BMI vs Obesity Risk'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'BMI'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Obesity Risk'
          }
        }
      }
    };

    return <Scatter data={data} options={options} />;
  };

  const renderLifestyleImpact = () => {
    if (!visualizationData?.lifestyle) return null;

    const data = {
      labels: visualizationData.lifestyle.diet_quality.map(d => d.quality),
      datasets: [{
        label: 'Average Obesity Risk',
        data: visualizationData.lifestyle.diet_quality.map(d => d.risk),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Diet Quality Impact on Obesity Risk'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Diet Quality Score'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Average Obesity Risk'
          }
        }
      }
    };

    return <Line data={data} options={options} />;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Data Insights
      </Typography>

      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Regional Distribution" />
        <Tab label="Health Metrics" />
        <Tab label="Lifestyle Impact" />
      </Tabs>

      <Box sx={{ height: 400 }}>
        {activeTab === 0 && renderRegionalDistribution()}
        {activeTab === 1 && renderHealthMetrics()}
        {activeTab === 2 && renderLifestyleImpact()}
      </Box>

      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Key Insights:
        </Typography>
        {activeTab === 0 && (
          <Typography variant="body2">
            • Analysis of obesity risk across Nigeria's geopolitical zones
            • Helps identify regions that may need targeted health interventions
            • Shows regional variations in health outcomes
          </Typography>
        )}
        {activeTab === 1 && (
          <Typography variant="body2">
            • Strong correlation between BMI and obesity risk
            • Higher BMI values generally indicate increased risk
            • Helps identify BMI thresholds for risk categories
          </Typography>
        )}
        {activeTab === 2 && (
          <Typography variant="body2">
            • Clear relationship between diet quality and obesity risk
            • Higher diet quality scores correlate with lower risk
            • Demonstrates the importance of dietary habits in obesity prevention
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

export default DataVisualization; 