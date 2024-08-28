import React, { useMemo } from 'react';
import { Box } from '@mui/material';

const createCurvedPath = (x1, y1, x2, y2) => {
  const midY = (y1 + y2) / 2;
  return `M ${x1} ${y1} Q ${x1} ${midY}, ${(x1 + x2) / 2} ${midY} T ${x2} ${y2}`;
};

const FlowComponent = ({ 
  data, 
  showInputs = true, 
  showOutputs = true, 
  tokens, 
  width, 
  height, 
  color="gray", 
  maxOpacity = 0.5, 
  useRelativeWeight = false,
  selectedInputs = new Set(),
  selectedOutputs = new Set(),
  onNodeClick = () => {}
}) => {
  const { inputs, flow, outputs } = data;
  const N = inputs.length;
  const circleRadius = 5;

  const inputPositions = useMemo(() => inputs.map((_, i) => ({
    x: (i + 1) * (width / (N + 1)),
    y: 20 + circleRadius * 2
  })), [inputs, width, N]);

  const outputPositions = useMemo(() => outputs.slice(0, -1).map((_, i) => ({
    x: (i + 1) * (width / (N + 1)),
    y: height - 20 - circleRadius * 2
  })), [outputs, width, height, N]);

  const maxFlow = useMemo(() => {
    if (!useRelativeWeight) {
      return 1; // Use absolute values
    }
    const relevantFlows = flow.flatMap((row, i) => 
      row.filter((_, j) => 
        (selectedInputs.size === 0 || selectedInputs.has(j)) &&
        (selectedOutputs.size === 0 || selectedOutputs.has(i))
      )
    );
    return relevantFlows.length > 0 ? Math.max(...relevantFlows) : Math.max(...flow.flat());
  }, [flow, useRelativeWeight, selectedInputs, selectedOutputs]);

  const shouldDrawConnection = (inputIndex, outputIndex) => {
    console.log(selectedInputs, selectedOutputs)
    return (selectedInputs.size === 0 || selectedInputs.has(inputIndex)) &&
           (selectedOutputs.size === 0 || selectedOutputs.has(outputIndex));
  };

  const paths = useMemo(() => {
    const result = [];
    for (let i = 0; i < flow.length - 1; i++) {
      for (let j = 0; j < flow[i].length; j++) {
        if (shouldDrawConnection(j, i)) {
          let strokeWidth = (flow[i][j] / maxFlow) * circleRadius * 2;
          
          result.push({
            key: `flow-${i}-${j}`,
            d: createCurvedPath(
              inputPositions[j].x,
              inputPositions[j].y,
              outputPositions[i].x,
              outputPositions[i].y
            ),
            strokeWidth: strokeWidth,
            inputIndex: j,
            outputIndex: i
          });
        }
      }
    }
    return result;
  }, [flow, inputPositions, outputPositions, maxFlow, selectedInputs, selectedOutputs]);

  return (
    <Box sx={{ width, height, position: 'relative' }}>
      <svg width={width} height={height}>
        <g>
          {paths.map(({ key, d, strokeWidth }) => (
            <path
              key={key}
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              opacity={maxOpacity}
            />
          ))}
        </g>
        {showInputs && inputPositions.map((pos, i) => (
          <g key={`input-${i}`}>
            <circle 
              cx={pos.x} 
              cy={pos.y} 
              r={circleRadius} 
              fill={selectedInputs.has(i) ? "green" : "blue"}
              onClick={(e) => onNodeClick(i, true, e)}
              style={{ cursor: 'pointer' }}
            />
            <text x={pos.x} y={pos.y - 10} textAnchor="middle" fontSize="12px">
              {tokens[i]}
            </text>
          </g>
        ))}
        {showOutputs && outputPositions.map((pos, i) => (
          <g key={`output-${i}`}>
            <circle 
              cx={pos.x} 
              cy={pos.y} 
              r={circleRadius} 
              fill={selectedOutputs.has(i) ? "green" : "red"}
              onClick={(e) => onNodeClick(i, false, e)}
              style={{ cursor: 'pointer' }}
            />
            <text x={pos.x} y={pos.y + 20} textAnchor="middle" fontSize="12px">
              {tokens[i + 1]}
            </text>
          </g>
        ))}
      </svg>
    </Box>
  );
};

export default FlowComponent;