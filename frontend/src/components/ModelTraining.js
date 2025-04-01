import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  LinearProgress
} from '@mui/material';
import { CloudUpload, Autorenew } from '@mui/icons-material';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const ModelTraining = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState({ status: '', message: '' });
  const [trainingStatus, setTrainingStatus] = useState({ status: '', message: '' });
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
      setUploadStatus({ status: '', message: '' });
    } else {
      setUploadStatus({ 
        status: 'error', 
        message: 'Please select a CSV file' 
      });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus({ 
        status: 'error', 
        message: 'Please select a file first' 
      });
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setUploadStatus({ 
        status: 'success', 
        message: `File uploaded successfully. ${response.data.rows} rows loaded.` 
      });
    } catch (error) {
      setUploadStatus({ 
        status: 'error', 
        message: error.response?.data?.detail || 'Error uploading file' 
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleRetrain = async () => {
    setIsTraining(true);
    setTrainingStatus({ status: '', message: '' });
    setTrainingMetrics(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/retrain`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log('Retraining response:', response.data);  // Debug log
      setTrainingStatus({ 
        status: 'success', 
        message: 'Model retrained successfully!' 
      });
      
      // Ensure we have valid numbers
      const accuracy = parseFloat(response.data.accuracy);
      const f1Score = parseFloat(response.data.f1_score);
      
      console.log('Parsed metrics:', { accuracy, f1Score });  // Debug log
      
      if (isNaN(accuracy) || isNaN(f1Score)) {
        console.error('Invalid metrics received:', response.data);
        throw new Error('Invalid metrics received from server');
      }
      
      const metrics = {
        accuracy: accuracy,
        f1_score: f1Score
      };
      
      console.log('Final training metrics:', metrics);  // Debug log
      setTrainingMetrics(metrics);
    } catch (error) {
      console.error('Retraining error:', error);  // Debug log
      const errorMessage = error.response?.data?.detail || error.message || 'Error retraining model';
      setTrainingStatus({ 
        status: 'error', 
        message: errorMessage
      });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Model Training
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Typography variant="subtitle1" gutterBottom>
          1. Upload Training Data
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Upload a CSV file containing new training data. The file must include all required columns:
          Region, Age, Gender, BMI, Blood Pressure (Systolic/Diastolic), Blood Sugar, Cholesterol,
          Physical Activity, Diet Quality, and Obesity Risk.
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Button
            variant="contained"
            component="label"
            startIcon={<CloudUpload />}
            disabled={isUploading}
          >
            Select CSV File
            <input
              type="file"
              accept=".csv"
              hidden
              onChange={handleFileSelect}
            />
          </Button>

          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
          >
            Upload
          </Button>
        </Box>

        {isUploading && <LinearProgress sx={{ mb: 2 }} />}
        
        {selectedFile && (
          <Typography variant="body2" sx={{ mb: 1 }}>
            Selected file: {selectedFile.name}
          </Typography>
        )}

        {uploadStatus.message && (
          <Alert severity={uploadStatus.status === 'success' ? 'success' : 'error'} sx={{ mb: 2 }}>
            {uploadStatus.message}
          </Alert>
        )}
      </Box>

      <Box>
        <Typography variant="subtitle1" gutterBottom>
          2. Retrain Model
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Retrain the model using the newly uploaded data. This will update the model's parameters
          and may improve its prediction accuracy.
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Button
            variant="contained"
            color="secondary"
            startIcon={<Autorenew />}
            onClick={handleRetrain}
            disabled={isTraining || uploadStatus.status !== 'success'}
          >
            {isTraining ? 'Training...' : 'Retrain Model'}
          </Button>
        </Box>

        {isTraining && <LinearProgress sx={{ mb: 2 }} />}

        {trainingStatus.message && (
          <Alert 
            severity={trainingStatus.status === 'success' ? 'success' : 'error'} 
            sx={{ mb: 2 }}
          >
            {typeof trainingStatus.message === 'string' ? trainingStatus.message : 'An error occurred during retraining'}
          </Alert>
        )}

        {trainingMetrics && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Training Results:
            </Typography>
            <Typography variant="body2">
              Accuracy: {Number(trainingMetrics.accuracy).toFixed(4)}
            </Typography>
            <Typography variant="body2">
              F1 Score: {Number(trainingMetrics.f1_score).toFixed(4)}
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default ModelTraining; 