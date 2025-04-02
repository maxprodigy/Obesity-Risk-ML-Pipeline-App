import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import axios from 'axios';

// API base URL
const API_URL = process.env.REACT_APP_API_URL;
console.log("API URL:", API_URL);

const PredictionForm = () => {
  const [features, setFeatures] = useState({
    region: 'North-Central',  // Default value for region
    bmi: '',
    age: '',
    gender: '',
    blood_pressure_systolic: '',
    blood_pressure_diastolic: '',
    blood_sugar: '',
    cholesterol: '',
    physical_activity: '',
    diet_quality: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      setApiStatus(response.data.status === 'ok' ? 'connected' : 'error');
    } catch (err) {
      console.error('API health check error:', err);
      setApiStatus('disconnected');
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFeatures(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    
    try {
      // Check API health first
      await checkApiHealth();
      
      if (apiStatus !== 'connected') {
        throw new Error('API is not available. Please check if the server is running.');
      }
      
      // Convert features to appropriate format for API
      const formattedFeatures = {
        ...features,
        bmi: parseFloat(features.bmi),
        age: parseInt(features.age),
        gender: features.gender === 'Male' ? 1 : 0,
        blood_pressure_systolic: parseInt(features.blood_pressure_systolic),
        blood_pressure_diastolic: parseInt(features.blood_pressure_diastolic),
        blood_sugar: parseFloat(features.blood_sugar),
        cholesterol: parseFloat(features.cholesterol),
        physical_activity: features.physical_activity === 'High' ? 2 : 
                         features.physical_activity === 'Moderate' ? 1 : 0,
        diet_quality: parseFloat(features.diet_quality)
      };
      
      const response = await axios.post(`${API_URL}/predict`, {
        features: formattedFeatures
      });
      
      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      
      if (err.response) {
        setError(`Server error: ${err.response.data.detail || err.response.statusText}`);
      } else if (err.request) {
        setError('No response from server. Please check if the API server is running.');
      } else {
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Enter Patient Information
        </Typography>
        <Box>
          <Typography variant="body2" component="span" sx={{ mr: 1 }}>
            API:
          </Typography>
          <Typography 
            variant="body2" 
            component="span" 
            sx={{ 
              color: apiStatus === 'connected' ? 'green' : 
                     apiStatus === 'checking' ? 'orange' : 'red',
              fontWeight: 'bold'
            }}
          >
            {apiStatus === 'connected' ? 'Connected' : 
             apiStatus === 'checking' ? 'Checking...' : 'Disconnected'}
          </Typography>
        </Box>
      </Box>

      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: 'repeat(2, 1fr)', mb: 3 }}>
          <FormControl fullWidth required>
            <InputLabel>Region</InputLabel>
            <Select
              name="region"
              value={features.region}
              onChange={handleChange}
              label="Region"
            >
              <MenuItem value="North-Central">North Central</MenuItem>
              <MenuItem value="Northeast">North East</MenuItem>
              <MenuItem value="Northwest">North West</MenuItem>
              <MenuItem value="South-South">South South</MenuItem>
              <MenuItem value="Southeast">South East</MenuItem>
              <MenuItem value="Southwest">South West</MenuItem>
              <MenuItem value="National">National</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            label="BMI"
            name="bmi"
            value={features.bmi}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <TextField
            label="Age"
            name="age"
            value={features.age}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <FormControl fullWidth required>
            <InputLabel>Gender</InputLabel>
            <Select
              name="gender"
              value={features.gender}
              onChange={handleChange}
              label="Gender"
            >
              <MenuItem value="Male">Male</MenuItem>
              <MenuItem value="Female">Female</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            label="Blood Pressure Systolic"
            name="blood_pressure_systolic"
            value={features.blood_pressure_systolic}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <TextField
            label="Blood Pressure Diastolic"
            name="blood_pressure_diastolic"
            value={features.blood_pressure_diastolic}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <TextField
            label="Blood Sugar"
            name="blood_sugar"
            value={features.blood_sugar}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <TextField
            label="Cholesterol"
            name="cholesterol"
            value={features.cholesterol}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
          
          <FormControl fullWidth required>
            <InputLabel>Physical Activity</InputLabel>
            <Select
              name="physical_activity"
              value={features.physical_activity}
              onChange={handleChange}
              label="Physical Activity"
            >
              <MenuItem value="Low">Low</MenuItem>
              <MenuItem value="Moderate">Moderate</MenuItem>
              <MenuItem value="High">High</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            label="Diet Quality"
            name="diet_quality"
            value={features.diet_quality}
            onChange={handleChange}
            type="number"
            fullWidth
            required
          />
        </Box>

        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          size="large"
          disabled={loading || apiStatus === 'disconnected'}
          sx={{ position: 'relative' }}
        >
          {loading ? (
            <>
              <CircularProgress 
                size={24} 
                sx={{ 
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  marginTop: '-12px',
                  marginLeft: '-12px'
                }} 
              />
              PREDICTING...
            </>
          ) : 'PREDICT RISK'}
        </Button>
      </form>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {prediction && (
        <Paper elevation={2} sx={{ mt: 3, p: 2, bgcolor: prediction.prediction === 1 ? '#ffebee' : '#e8f5e9' }}>
          <Typography variant="h6" gutterBottom>
            Prediction Result
          </Typography>
          <Typography variant="body1">
            Risk Level: {prediction.message}
          </Typography>
          <Typography variant="body1">
            Probability: {(prediction.probability * 100).toFixed(2)}%
          </Typography>
        </Paper>
      )}
    </Paper>
  );
};

export default PredictionForm; 
