// src/components/ImageDisplay.js
import React, { useState, useEffect } from 'react';
import HeatMapGrid from 'react-heatmap-grid';

function ImageDisplay({ matrix }) {
  // State to manage cell size
  const [cellSize, setCellSize] = useState(20);

  if (!matrix || !matrix.length) return <p>No data available.</p>;

  const rows = matrix.length;
  const cols = matrix[0].length;

  // Generate row and column labels (optional)
  const xLabels = new Array(cols).fill('').map((_, i) => `${i}`);
  const yLabels = new Array(rows).fill('').map((_, i) => `${i}`);

  // Find the minimum and maximum values in the matrix
  const flatValues = matrix.flat();
  const minValue = Math.min(...flatValues);
  const maxValue = Math.max(...flatValues);

  // Normalize the data for the heatmap
  const normalizeValue = (value) => (value - minValue) / (maxValue - minValue);

  const normalizedMatrix = matrix.map((row) => row.map(normalizeValue));

  return (
    <div>
      <h2>Generated Heatmap:</h2>
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="cell-size-slider">
          Cell Size: {cellSize}px
        </label>
        <input
          id="cell-size-slider"
          type="range"
          min="10"
          max="50"
          value={cellSize}
          onChange={(e) => setCellSize(Number(e.target.value))}
          style={{ marginLeft: '10px', verticalAlign: 'middle' }}
        />
      </div>
      <HeatMapGrid
        xLabels={xLabels}
        yLabels={yLabels}
        data={normalizedMatrix}
        xLabelsStyle={(index) => ({
          color: '#777',
          fontSize: '12px',
        })}
        yLabelsStyle={() => ({
          color: '#777',
          fontSize: '12px',
        })}
        cellStyle={(background, value, min, max, data, x, y) => ({
            background: `rgba(66, 86, 244, ${1 - (max - value) / (max - min)})`,
            fontSize: "11px",
          })}
        cellRender={(value) => value && `${value.toFixed(3)}`}
        squares
      />
    </div>
  );
}

export default ImageDisplay;
