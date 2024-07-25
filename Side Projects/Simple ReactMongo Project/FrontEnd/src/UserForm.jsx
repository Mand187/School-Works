import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Box, Typography, CircularProgress, Snackbar, Alert } from '@mui/material';

const UserForm = ({ onUserAdded }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:3000/users/create', {
        username,
        password,
      });
      onUserAdded(response.data);
      setUsername('');
      setPassword('');
    } catch (error) {
      setError('There was an error creating the user!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 400, width: '100%', padding: 2, boxShadow: 3, borderRadius: 2, backgroundColor: '#fff' }}>
        <Typography variant="h5" component="h1" gutterBottom>
          Create User
        </Typography>
        <TextField
          label="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          fullWidth
          margin="normal"
        />
        <TextField
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          fullWidth
          margin="normal"
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          sx={{ mt: 2 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Create User'}
        </Button>
        {error && (
          <Snackbar open={Boolean(error)} autoHideDuration={6000} onClose={() => setError(null)}>
            <Alert onClose={() => setError(null)} severity="error">
              {error}
            </Alert>
          </Snackbar>
        )}
      </Box>
    </Box>
  );
};

export default UserForm;
