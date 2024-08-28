// src/components/Loader.js
import React from 'react';
import { CircularProgress, Box } from '@mui/material';

function Loader({ loading }) {
  return (
    loading && (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    )
  );
}

export default Loader;
