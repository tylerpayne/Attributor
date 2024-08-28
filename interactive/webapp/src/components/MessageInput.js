import React from 'react';
import { TextField, Select, MenuItem, Box, FormControl, InputLabel } from '@mui/material';

const MessageInput = ({ role, content, onRoleChange, onContentChange }) => {
  return (
    <Box sx={{ display: 'flex', '& > :not(style)': { m: 1 } }}>
      <FormControl sx={{ minWidth: 120 }}>
        <InputLabel>Role</InputLabel>
        <Select
          value={role}
          onChange={(e) => onRoleChange(e.target.value)}
          label="Role"
        >
          <MenuItem value="system">System</MenuItem>
          <MenuItem value="user">User</MenuItem>
          <MenuItem value="assistant">Assistant</MenuItem>
        </Select>
      </FormControl>
      <TextField
        label="Message content"
        variant="outlined"
        fullWidth
        value={content}
        onChange={(e) => onContentChange(e.target.value)}
      />
    </Box>
  );
};

export default MessageInput;