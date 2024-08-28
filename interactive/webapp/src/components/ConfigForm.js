// src/components/ConfigForm.js
import React, { useState } from 'react';
import { TextField, Button, Box } from '@mui/material';

function ConfigForm({ onSubmit, loading }) {
  const [config, setConfig] = useState({
    model: '',
    device_map: '',
    dtype: '',
    max_context_tokens: 0,
  });

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(config);
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
      <TextField
        label="Model"
        variant="outlined"
        value={config.model}
        onChange={(e) => setConfig({ ...config, model: e.target.value })}
        required
        fullWidth
        sx={{ mb: 2 }}
      />
      <TextField
        label="Device Map"
        variant="outlined"
        value={config.device_map}
        onChange={(e) => setConfig({ ...config, device_map: e.target.value })}
        required
        fullWidth
        sx={{ mb: 2 }}
      />
      <TextField
        label="Dtype"
        variant="outlined"
        value={config.dtype}
        onChange={(e) => setConfig({ ...config, dtype: e.target.value })}
        required
        fullWidth
        sx={{ mb: 2 }}
      />
      <TextField
        label="Max Context Tokens"
        variant="outlined"
        type="number"
        value={config.max_context_tokens}
        onChange={(e) =>
          setConfig({ ...config, max_context_tokens: parseInt(e.target.value) })
        }
        required
        fullWidth
        sx={{ mb: 2 }}
      />
      <Button variant="contained" color="primary" type="submit" disabled={loading}>
        {loading ? 'Configuring...' : 'Submit Configuration'}
      </Button>
    </Box>
  );
}

export default ConfigForm;
