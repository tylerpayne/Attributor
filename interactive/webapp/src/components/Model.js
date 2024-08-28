import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Switch,
  FormControlLabel,
} from '@mui/material';
import axios from 'axios';
import FlowComponent from './FlowComponent';
import Layer from './Layer';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000'
});

const Model = ({ modelParams }) => {
  const [modelData, setModelData] = useState(null);
  const [activeLayers, setActiveLayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [toggleAll, setToggleAll] = useState(false);
  const [dimensions, setDimensions] = useState({
    width: modelParams.tokens.length * 50,
    height: 500,
  });

  const containerRef = useRef(null)

  useEffect(() => {
    const fetchModelData = async () => {
      setLoading(true);
      try {
        const [inputs, flow, outputs] = await Promise.all([
          api.get('/api/v1/model/inputs'),
          api.get('/api/v1/model/flow'),
          api.get('/api/v1/model/outputs')
        ]);
        setModelData({
          inputs: inputs.data,
          flow: flow.data,
          outputs: outputs.data,
        });
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchModelData();
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
          setDimensions({
            width: containerRef.current.offsetWidth,
            height: containerRef.current.offsetHeight,
          });
      }
    };

    if (containerRef.current) {
      handleResize()
    }

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleLayerToggle = (event, newActiveLayers) => {
    newActiveLayers.sort()
    setActiveLayers(newActiveLayers);
    setToggleAll(newActiveLayers.length === modelParams.nr_layers);
  };

  const handleToggleAll = (event) => {
    const newToggleAll = event.target.checked;
    setToggleAll(newToggleAll);
    setActiveLayers(newToggleAll ? [...Array(modelParams.nr_layers).keys()] : []);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography color="error">Error: {error}</Typography>
      </Box>
    );
  }

  if (!modelData) {
    return null;
  }

  return (
    <Paper elevation={3} sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        Model Overview
      </Typography>

      <Box sx={{ mb: 2, overflowX: 'auto' }}>
        <Typography variant="h6" gutterBottom>
          Layers
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={toggleAll}
              onChange={handleToggleAll}
              name="toggleAll"
            />
          }
          label="Toggle All Layers"
        />
        <ToggleButtonGroup
          value={activeLayers}
          onChange={handleLayerToggle}
          aria-label="active layers"
        >
          {[...Array(modelParams.nr_layers)].map((_, index) => (
            <ToggleButton 
              key={index} 
              value={index} 
              aria-label={`toggle layer ${index + 1}`}
            >
              Layer {index + 1}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      <Box ref={containerRef} sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 2, overflowY: 'auto' }}>
        {activeLayers.length === 0 ? (
          <FlowComponent 
            data={modelData} 
            tokens={modelParams.tokens} 
            width={dimensions.width}
            height={dimensions.height}
          />
        ) : (
          activeLayers.map((layerIndex, index) => (
            <Layer
              key={layerIndex}
              layerIndex={layerIndex}
              isStacked={activeLayers.length > 1}
              isFirst={index === 0}
              isLast={index === activeLayers.length - 1}
              nrAttentionHeads={modelParams.nr_attention_heads}
              tokens={modelParams.tokens}
            />
          ))
        )}
      </Box>

      <Box sx={{ mt: 2 }}>
        <Typography variant="body2">
          Total Layers: {modelParams.nr_layers}
        </Typography>
        <Typography variant="body2">
          Attention Heads per Layer: {modelParams.nr_attention_heads}
        </Typography>
        <Typography variant="body2">
          Tokens: {modelParams.tokens.join(', ')}
        </Typography>
      </Box>
    </Paper>
  );
};

export default Model;