import React from 'react';
import { Box } from '@mui/material';

export const TokenDistribution = ({ distribution, tokens, x, y, width, height }) => {
  const barWidth = width / distribution.length;
  const maxValue = Math.max(...distribution);

  return (
    <g transform={`translate(${x}, ${y})`}>
      {distribution.map((value, i) => (
        <rect
          key={i}
          x={i * barWidth}
          y={height - (value / maxValue) * height}
          width={barWidth}
          height={(value / maxValue) * height}
          fill="rgba(0, 0, 255, 0.5)"
        />
      ))}
      {tokens.map((token, i) => (
        <text
          key={i}
          x={(i + 0.5) * barWidth}
          y={height + 15}
          textAnchor="middle"
          fontSize="10"
          transform={`rotate(45, ${(i + 0.5) * barWidth}, ${height + 15})`}
        >
          {token}
        </text>
      ))}
    </g>
  );
};

export const WeightDistribution = ({ weights, width, height }) => {
  const maxWeight = Math.max(...weights);

  return (
    <Box component="svg" width={width} height={height}>
      {weights.map((weight, i) => {
        const barHeight = (weight / maxWeight) * height;
        const barWidth = width / weights.length;
        return (
          <g key={i}>
            <rect
              x={i * barWidth}
              y={height - barHeight}
              width={barWidth}
              height={barHeight}
              fill="rgba(0, 128, 0, 0.5)"
            />
            <text
              x={(i + 0.5) * barWidth}
              y={height + 15}
              textAnchor="middle"
              fontSize="10"
            >
              {weight.toFixed(2)}
            </text>
          </g>
        );
      })}
    </Box>
  );
};