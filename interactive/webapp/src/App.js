// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Typography } from '@mui/material';
import ConfigForm from './components/ConfigForm';
import MessagesForm from './components/MessagesForm';
import ImageDisplay from './components/ImageDisplay';
import Loader from './components/Loader';
import Model from './components/Model'

function App() {
  const [isConfigured, setIsConfigured] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelParams, setModelParams] = useState(null);

  const checkConfiguration = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/configured');
      const data = await response.json();
      setIsConfigured(data);
    } catch (error) {
      console.error('Error checking configuration:', error);
    }
  };

  const handleConfigSubmit = async (config) => {
    setLoading(true);
    try {
      console.log(config)
      const response = await fetch('http://127.0.0.1:8000/api/v1/configure', {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        alert('Configuration successful!');
      } else {
        alert('Configuration failed. Please check your input.');
      }
    } catch (error) {
      console.error('Error configuring model:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAttributionSubmit = async (messages) => {
    console.log(messages)
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/v1/attribute', messages, {
        headers: { 'Content-Type': 'application/json' },
      });
      console.log(response.data)
      setModelParams(response.data);
    } catch (error) {
      console.error('Error submitting messages:', error);
    }
    finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      checkConfiguration();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom>
        Text Attribution Visualization
      </Typography>
      <Typography variant="h6" component="h2" gutterBottom>
        Model Configuration
      </Typography>
      <ConfigForm onSubmit={handleConfigSubmit} loading={loading} />
      {
        isConfigured && 
        <>
          <Typography variant="h6" component="h2" gutterBottom sx={{ mt: 4 }}>
            Enter Text for Attribution
          </Typography>
          <MessagesForm onSubmit={handleAttributionSubmit} loading={loading} disabled={!isConfigured} />
          {modelParams && <Model modelParams={modelParams} />}
          <Loader loading={loading} /> 
        </>
      }
    </Container>
  );
}

export default App;
