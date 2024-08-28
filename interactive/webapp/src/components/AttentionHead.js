import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import axios from 'axios';
import FlowComponent from './FlowComponent';

const AttentionHead = ({ 
  layerIndex, 
  headIndex,
  showInputs = true,
  showOutputs = true,
  tokens,
  width,
  height,
  color = "gray",
  maxOpacity = 0.5,
  useRelativeWeight,
  selectedInputs,
  selectedOutputs,
  onNodeClick }) => {
  const [headData, setHeadData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHeadData = async () => {
      setLoading(true);
      try {
        const [inputs, flow, outputs] = await Promise.all([
          axios.get(`http://127.0.0.1:8000/api/v1/model/layers/${layerIndex}/attention_heads/${headIndex}/inputs`),
          axios.get(`http://127.0.0.1:8000/api/v1/model/layers/${layerIndex}/attention_heads/${headIndex}/flow`),
          axios.get(`http://127.0.0.1:8000/api/v1/model/layers/${layerIndex}/attention_heads/${headIndex}/outputs`),
        ]);

        setHeadData({
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

    fetchHeadData();
  }, [layerIndex, headIndex]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="error">Error: {error}</Typography>
      </Box>
    );
  }

  if (!headData) {
    return null;
  }

  return (
    <FlowComponent 
      data={headData}                 
      showInputs={showInputs}
      showOutputs={showOutputs}
      width={width}
      maxOpacity={maxOpacity}
      color={color}
      tokens={tokens}
      height={height}
      useRelativeWeight={useRelativeWeight}
      selectedInputs={selectedInputs}
      selectedOutputs={selectedOutputs}
      onNodeClick={onNodeClick} 
    />
  )
};

export default AttentionHead;