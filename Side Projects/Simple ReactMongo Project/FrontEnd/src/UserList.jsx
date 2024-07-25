import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { CircularProgress, Box, Typography, Snackbar, Alert } from '@mui/material';

const UserList = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await axios.get('http://localhost:3000/users/all');
        setUsers(response.data);
      } catch (error) {
        setError('There was an error fetching the users!');
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h4" component="h2" gutterBottom>
        User List
      </Typography>
      {loading ? (
        <CircularProgress />
      ) : error ? (
        <Snackbar open={Boolean(error)} autoHideDuration={6000} onClose={() => setError(null)}>
          <Alert onClose={() => setError(null)} severity="error">
            {error}
          </Alert>
        </Snackbar>
      ) : (
        <ul>
          {users.length > 0 ? (
            users.map((user) => (
              <li key={user._id}>{user.username}</li>
            ))
          ) : (
            <Typography>No users found</Typography>
          )}
        </ul>
      )}
    </Box>
  );
};

export default UserList;
