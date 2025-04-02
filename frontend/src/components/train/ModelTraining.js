import React, { useState } from 'react';
import {
  Paper,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  LinearProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ModelTraining = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileInfo, setFileInfo] = useState(null);

  // Feature name mapping
  const featureNames = {
    'feature_0': 'BMI',
    'feature_1': 'Age',
    'feature_2': 'Gender',
    'feature_3': 'Physical Activity'
  };
  
  const featureDescriptions = {
    'BMI': 'Body Mass Index (weight/height²)',
    'Age': 'Patient Age in Years',
    'Gender': 'Male/Female Classification',
    'Physical Activity': 'Exercise Frequency Level'
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
      setError(null);
      // Display file info
      setFileInfo({
        name: file.name,
        size: (file.size / 1024).toFixed(2) + ' KB',
      });
    } else {
      setError('Please select a valid CSV file');
      setSelectedFile(null);
      setFileInfo(null);
    }
  };

  const handleUploadAndTrain = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setTrainingStatus('Uploading file...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // First, validate the file
      setTrainingStatus('Validating file...');
      const validateResponse = await fetch(`${process.env.REACT_APP_API_URL}/validate`, {
        method: 'POST',
        body: formData,
      });

      if (!validateResponse.ok) {
        const validateData = await validateResponse.json();
        throw new Error(validateData.detail || 'Invalid file format');
      }

      const validateResult = await validateResponse.json();
      setTrainingStatus(`File validated. Found ${validateResult.rows} rows.`);
      
      // Show retraining message for at least 10 seconds
      setTrainingStatus('Retraining model...');
      
      // Track the start time
      const startTime = Date.now();
      
      // Retrain the model
      const response = await fetch('http://localhost:8000/retrain', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        // Handle specific error for missing columns
        if (errorData.detail && errorData.detail.includes('missing required columns')) {
          throw new Error(`Dataset Validation Error: ${errorData.detail}`);
        }
        throw new Error(errorData.detail || 'Training failed');
      }

      const data = await response.json();
      
      // Calculate elapsed time
      const elapsedTime = Date.now() - startTime;
      const remainingTime = Math.max(0, 10000 - elapsedTime);
      
      // If less than 10 seconds have passed, wait the remaining time
      if (remainingTime > 0) {
        setTrainingStatus('Finalizing model training...');
        await new Promise(resolve => setTimeout(resolve, remainingTime));
      }
      
      setTrainingStatus('Training completed successfully');
      setTrainingMetrics(data);
    } catch (err) {
      setError(err.message || 'An error occurred during training');
      setTrainingStatus('Training failed');
    } finally {
      setIsLoading(false);
    }
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
      <Box 
        sx={{ 
          border: '2px dashed #FFD700',
          borderRadius: 2,
          p: 3,
          textAlign: 'center',
          mb: 3,
          cursor: 'pointer',
          '&:hover': {
            bgcolor: 'rgba(255, 215, 0, 0.1)',
          },
        }}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          type="file"
          id="file-input"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
        <CloudUploadIcon sx={{ fontSize: 48, color: '#FFD700', mb: 2 }} />
        <Typography variant="h6" gutterBottom sx={{ color: '#FFD700' }}>
          Drop CSV file here or click to upload
        </Typography>
        {fileInfo ? (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body1" sx={{ color: '#FFD700' }}>
              Selected file: {fileInfo.name}
            </Typography>
            <Typography variant="body2" sx={{ color: '#FFD700', opacity: 0.8 }}>
              Size: {fileInfo.size}
            </Typography>
          </Box>
        ) : (
          <Typography variant="body2" sx={{ color: '#FFD700', opacity: 0.8 }}>
            Supported format: CSV
          </Typography>
        )}
      </Box>

      <Button
        variant="contained"
        fullWidth
        disabled={!selectedFile || isLoading}
        onClick={handleUploadAndTrain}
        sx={{
          bgcolor: '#000000',
          color: '#FFD700',
          border: '2px solid #FFD700',
          '&:hover': {
            bgcolor: '#FFD700',
            color: '#000000',
          },
          height: 48,
          fontSize: '1.1rem',
          fontWeight: 'bold',
          mb: 3,
        }}
      >
        {isLoading ? <CircularProgress size={24} sx={{ color: '#FFD700' }} /> : 'Train Model'}
      </Button>

      {isLoading && (
        <Box sx={{ width: '100%', mb: 3 }}>
          <LinearProgress 
            sx={{
              bgcolor: 'rgba(255, 215, 0, 0.1)',
              '& .MuiLinearProgress-bar': {
                bgcolor: '#FFD700',
              },
            }}
          />
          <Typography variant="body2" sx={{ color: '#FFD700', mt: 1 }}>
            {trainingStatus}
          </Typography>
        </Box>
      )}

      {error && (
        <Alert 
          severity="error"
          sx={{
            bgcolor: 'transparent',
            color: '#FFD700',
            border: '1px solid #FFD700',
            mb: 3,
            '& .MuiAlert-icon': {
              color: '#FFD700',
            },
          }}
        >
          <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
            {error}
          </Typography>
        </Alert>
      )}

      {trainingMetrics && (
        <Paper 
          sx={{ 
            p: 3, 
            bgcolor: '#000000',
            border: '1px solid #FFD700',
            borderRadius: 2,
          }}
        >
          <Typography variant="h6" gutterBottom sx={{ color: '#FFD700' }}>
            Training Results
          </Typography>
          <Typography variant="body1" sx={{ color: '#FFD700', mb: 1 }}>
            Accuracy: {(trainingMetrics.accuracy * 100).toFixed(2)}%
          </Typography>
          <Typography variant="body1" sx={{ color: '#FFD700', mb: 1 }}>
            F1 Score: {(trainingMetrics.f1_score * 100).toFixed(2)}%
          </Typography>
          {trainingMetrics.feature_importance && (
            <>
              <Typography variant="h6" sx={{ color: '#FFD700', mt: 3, mb: 2 }}>
                Feature Importance
              </Typography>
              {Object.entries(trainingMetrics.feature_importance)
                .sort(([, a], [, b]) => b - a)
                .map(([feature, importance]) => {
                  const featureName = featureNames[feature] || feature;
                  const description = featureDescriptions[featureName] || '';
                  return (
                    <Box key={feature} sx={{ mb: 1.5 }}>
                      <Typography variant="body2" sx={{ color: '#FFD700', fontWeight: 'bold' }}>
                        {featureName}: {(importance * 100).toFixed(2)}%
                      </Typography>
                      {description && (
                        <Typography variant="caption" sx={{ color: '#FFD700', opacity: 0.7, ml: 1 }}>
                          {description}
                        </Typography>
                      )}
                    </Box>
                  );
                })}
            </>
          )}
        </Paper>
      )}
    </Paper>
  );
};

export default ModelTraining; 
