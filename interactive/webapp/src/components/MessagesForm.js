// src/components/AttributionForm.js
import React, { useState } from 'react';
import { Typography, Button, Box, IconButton } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import MessageInput from './MessageInput'

function MessagesForm({ onSubmit, loading, disabled }) {
  const [messages, setMessages] = useState([]);

  const addMessage = () => {
    setMessages([...messages, { id: Date.now(), role: 'user', content: '' }]);
  };

  const updateMessage = (id, field, value) => {
    setMessages(messages.map(message => 
      message.id === id ? { ...message, [field]: value } : message
    ));
  };

  const removeMessage = (id) => {
    setMessages(messages.filter(message => message.id !== id));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!disabled) {
      const validMessages = messages.filter(message => message.content.trim() !== '');
      onSubmit(validMessages);
    }
  };

  return (
    <Box mt={3}>
      <Typography variant="h6" gutterBottom>
        Messages:
      </Typography>
      {messages.map((message) => (
        <Box key={message.id} display="flex" alignItems="center" mb={2}>
          <Box flexGrow={1}>
            <MessageInput
              role={message.role}
              content={message.content}
              onRoleChange={(value) => updateMessage(message.id, 'role', value)}
              onContentChange={(value) => updateMessage(message.id, 'content', value)}
            />
          </Box>
          <IconButton onClick={() => removeMessage(message.id)}>
            <DeleteIcon />
          </IconButton>
        </Box>
      ))}
      <Button
        onClick={addMessage}
        variant="outlined"
        color="primary"
        fullWidth
        sx={{ mt: 2, mb: 2 }}
      >
        Add Message
      </Button>
      <Box mt={4} mb={2}>
        <Button 
          onClick={handleSubmit} 
          disabled={messages.length === 0 || disabled}
          variant="contained" 
          color="primary"
          fullWidth
          size="large"
        >
          Attribute
        </Button>
      </Box>
    </Box>
  );
}

export default MessagesForm;