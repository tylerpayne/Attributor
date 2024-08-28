import React, { useState, useEffect, useRef, useCallback, useLayoutEffect } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material';
import axios from 'axios';
import FlowComponent from './FlowComponent';
import AttentionHead from './AttentionHead';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000'
});

const Layer = ({ layerIndex, isStacked, isFirst, isLast, nrAttentionHeads, tokens }) => {
  const [layerData, setLayerData] = useState(null);
  const [visibleHeads, setVisibleHeads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [toggleAll, setToggleAll] = useState(false);
  const [useRelativeWeight, setUseRelativeWeight] = useState(false);
  const [selectedInputs, setSelectedInputs] = useState(new Set());
  const [selectedOutputs, setSelectedOutputs] = useState(new Set());
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: 200,
  });

  const fetchLayerData = async () => {
    setLoading(true);
    try {
      const [inputs, flow, outputs, attentionHeadWeights] = await Promise.all([
        api.get(`/api/v1/model/layers/${layerIndex}/inputs`),
        api.get(`/api/v1/model/layers/${layerIndex}/flow`),
        api.get(`/api/v1/model/layers/${layerIndex}/outputs`),
        api.get(`/api/v1/model/layers/${layerIndex}/attention_head_weights`),
      ]);

      setLayerData({
        inputs: inputs.data,
        flow: flow.data,
        outputs: outputs.data,
        attention_head_weights: attentionHeadWeights.data,
      });
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLayerData();
  }, [layerIndex]);

  const updateDimensions = useCallback((node) => {
    if (node !== null) {
      const { width, height } = node.getBoundingClientRect();
      setDimensions({ width, height });
    }
  }, []);

  const handleHeadToggle = (event, newVisibleHeads) => {
    setVisibleHeads(newVisibleHeads);
    setToggleAll(newVisibleHeads.length === nrAttentionHeads);
  };

  const handleToggleAll = (event) => {
    const newToggleAll = event.target.checked;
    setToggleAll(newToggleAll);
    setVisibleHeads(newToggleAll ? [...Array(nrAttentionHeads).keys()] : []);
  };

  const handleRelativeWeightToggle = (event) => {
    setUseRelativeWeight(event.target.checked);
  };

  const handleNodeClick = (index, isInput, event) => {
    const stateUpdater = isInput ? setSelectedInputs : setSelectedOutputs;
    const otherStateUpdater = isInput ? setSelectedOutputs : setSelectedInputs;

    if (event.shiftKey) {
      stateUpdater(prevSelected => {
        const newSelected = new Set(prevSelected);
        if (newSelected.has(index)) {
          newSelected.delete(index);
        } else {
          newSelected.add(index);
        }
        return newSelected;
      });
    } else {
      stateUpdater(prevSelected => {
        const newSelected = new Set();
        if (!prevSelected.has(index)) {
          newSelected.add(index);
        }
        return newSelected;
      });
      otherStateUpdater(new Set());
    }
  };

  const getColorForHead = (index) => {
    const hue = (index * 137) % 360;
    return `hsla(${hue}, 70%, 50%, 1)`;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <Typography color="error">Error: {error}</Typography>
      </Box>
    );
  }

  if (!layerData) {
    return null;
  }

  const contentWidth = Math.max(dimensions.width, tokens.length * 50);

  
  return (
    <Box 
      sx={{ 
      display: 'flex', 
      flexDirection: 'row', 
      border: '1px solid #ccc', 
      borderRadius: '4px',
      overflow: 'hidden', // Prevent content from overflowing
    }}>
      <Box 
        ref={updateDimensions}
        sx={{ 
        width: '200px', 
        p: 2, 
        borderRight: '1px solid #ccc', 
        display: 'flex', 
        flexDirection: 'column',
        overflowY: 'auto', // Allow scrolling if content exceeds height
      }}>
        <Typography variant="h6" gutterBottom>
          Layer {layerIndex + 1}
        </Typography>
        
        <FormControlLabel
          control={
            <Switch
              checked={toggleAll}
              onChange={handleToggleAll}
              name="toggleAll"
              color="primary"
            />
          }
          label="Toggle All Heads"
          sx={{ mb: 1 }}
        />

        <FormControlLabel
          control={
            <Switch
              checked={useRelativeWeight}
              onChange={handleRelativeWeightToggle}
              name="useRelativeWeight"
              color="primary"
            />
          }
          label="Use Relative Weight"
          sx={{ mb: 2 }}
        />
        
        <Typography variant="subtitle2" gutterBottom>
          Attention Heads
        </Typography>
        
        <ToggleButtonGroup
          size="small"
          value={visibleHeads}
          onChange={handleHeadToggle}
          aria-label="visible attention heads"
          orientation="vertical"
        >
          {[...Array(nrAttentionHeads)].map((_, index) => (
            <ToggleButton 
              key={index} 
              value={index} 
              aria-label={`toggle head ${index + 1}`}
              style={{ 
                borderColor: getColorForHead(index),
                color: visibleHeads.includes(index) ? getColorForHead(index) : 'inherit',
                // opacity: getNormalizedWeight(layerData.attention_head_weights[index])
              }}
            >
              Head {index + 1}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>
      
      <Box 
        sx={{ 
        flex: 1, 
        position: 'relative', 
        overflowX: 'auto',
        overflowY: 'hidden',
      }}>
        <Box 
        sx={{ 
          width: contentWidth, 
          height: '100%', 
          position: 'relative',
        }}>
          {visibleHeads.length === 0 && 
            <FlowComponent 
              data={layerData} 
              showInputs={true}
              showOutputs={true}
              tokens={tokens}
              width={contentWidth}
              height={dimensions.height}
              useRelativeWeight={useRelativeWeight}
              selectedInputs={selectedInputs}
              selectedOutputs={selectedOutputs}
              onNodeClick={handleNodeClick}
            />
          }
          {visibleHeads.map((headIndex, index) => (
            <Box key={headIndex} sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
              <AttentionHead
                layerIndex={layerIndex}
                headIndex={headIndex}
                color={getColorForHead(headIndex)}
                showInputs={index == visibleHeads.length - 1}
                showOutputs={index == visibleHeads.length - 1}
                tokens={tokens}
                width={contentWidth}
                height={dimensions.height}
                useRelativeWeight={useRelativeWeight}
                selectedInputs={selectedInputs}
                selectedOutputs={selectedOutputs}
                onNodeClick={handleNodeClick}
              />
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default Layer;