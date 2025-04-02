import React, { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Grid,
  Typography,
  MenuItem,
  Box,
  Alert,
  CircularProgress
} from '@mui/material';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    age: '',
    gender: 'male',
    height: '',
    weight: '',
    physical_activity: 'medium'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Convert form data to numeric values where needed
      const payload = {
        age: parseInt(formData.age),
        gender: formData.gender,
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        physical_activity: formData.physical_activity
      };
      
      console.log("Sending prediction request:", payload);
      
      const response = await fetch(`${process.env.REACT_APP_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error("Prediction error response:", errorData);
        throw new Error(`Failed to get prediction: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Prediction response:", data);
      setPrediction(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(`Failed to get prediction. Please try again. (${err.message})`);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const textFieldStyle = {
    '& .MuiOutlinedInput-root': {
      '& fieldset': {
        borderColor: '#FFD700',
      },
      '&:hover fieldset': {
        borderColor: '#FFD700',
      },
      '&.Mui-focused fieldset': {
        borderColor: '#FFD700',
      },
    },
    '& .MuiInputLabel-root': {
      color: '#FFD700',
    },
    '& .MuiInputBase-input': {
      color: '#FFD700',
    },
    '& .MuiSelect-icon': {
      color: '#FFD700',
    },
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 4, 
        bgcolor: '#000000',
        border: '1px solid #FFD700',
        borderRadius: 2,
      }}
    >
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ color: '#FFD700', mb: 2 }}>
              Obesity Risk Prediction
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Age"
              name="age"
              type="number"
              value={formData.age}
              onChange={handleChange}
              required
              sx={textFieldStyle}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              select
              label="Gender"
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
              sx={textFieldStyle}
            >
              <MenuItem value="male">Male</MenuItem>
              <MenuItem value="female">Female</MenuItem>
            </TextField>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Height (cm)"
              name="height"
              type="number"
              value={formData.height}
              onChange={handleChange}
              required
              sx={textFieldStyle}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Weight (kg)"
              name="weight"
              type="number"
              value={formData.weight}
              onChange={handleChange}
              required
              sx={textFieldStyle}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              select
              label="Physical Activity"
              name="physical_activity"
              value={formData.physical_activity}
              onChange={handleChange}
              required
              sx={textFieldStyle}
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
            </TextField>
          </Grid>

          <Grid item xs={12}>
            <Button 
              variant="contained" 
              type="submit"
              disabled={loading}
              sx={{ 
                bgcolor: '#FFD700', 
                color: '#000000',
                '&:hover': {
                  bgcolor: '#E6C300',
                },
                mt: 2
              }}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict Risk'}
            </Button>
          </Grid>
          
          {error && (
            <Grid item xs={12}>
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            </Grid>
          )}
        </Grid>
      </form>

      {prediction && (
        <Paper 
          sx={{ 
            mt: 3,
            p: 3, 
            bgcolor: '#000000',
            border: '1px solid #FFD700',
            borderRadius: 2,
          }}
        >
          <Typography variant="h6" gutterBottom sx={{ color: '#FFD700' }}>
            Prediction Result
          </Typography>
          <Typography variant="body1" sx={{ color: '#FFD700' }}>
            Risk Level: {prediction.risk_level}
          </Typography>
          <Typography variant="body2" sx={{ color: '#FFD700', opacity: 0.8, mt: 1 }}>
            Confidence: {(prediction.probabilities[prediction.risk_level] * 100).toFixed(2)}%
          </Typography>
          
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255, 215, 0, 0.3)' }}>
            <Typography variant="subtitle2" sx={{ color: '#FFD700', mb: 1 }}>
              Probability Distribution:
            </Typography>
            {Object.entries(prediction.probabilities).map(([level, prob]) => (
              <Typography key={level} variant="body2" sx={{ color: '#FFD700', opacity: 0.7 }}>
                {level}: {(prob * 100).toFixed(2)}%
              </Typography>
            ))}
          </Box>
        </Paper>
      )}
    </Paper>
  );
};

export default PredictionForm; 
